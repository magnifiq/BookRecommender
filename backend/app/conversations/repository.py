import json
import sqlite3
from uuid import uuid4
from pathlib import Path
from typing import Protocol

from app.conversations.models import Conversation, ConversationMessage, Preference


class ConversationStorageError(RuntimeError):
    """A storage operation failed; callers should not expose its details."""


class ConversationRepository(Protocol):
    def create(self, conversation: Conversation) -> Conversation: ...
    def get(self, conversation_id: str) -> Conversation | None: ...
    def list_conversations(self, limit: int = 100) -> list[Conversation]: ...
    def delete(self, conversation_id: str) -> bool: ...
    def append(self, message: ConversationMessage) -> ConversationMessage: ...
    def messages(self, conversation_id: str) -> list[ConversationMessage]: ...
    def save_preference(self, preference: Preference) -> Preference: ...
    def preferences(self, conversation_id: str) -> list[Preference]: ...
    def delete_preference(self, conversation_id: str, preference_id: str) -> bool: ...
    def clear_preferences(self, conversation_id: str) -> int: ...


class SQLiteConversationRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        self._memory_connection = None
        if self.database_path == ":memory:":
            self._dsn = f"file:conversation-{uuid4()}?mode=memory&cache=shared"
            self._uri = True
            self._memory_connection = sqlite3.connect(self._dsn, uri=True, timeout=5)
            self._memory_connection.row_factory = sqlite3.Row
        else:
            self._dsn = self.database_path
            self._uri = False
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except sqlite3.Error as exc:
            raise ConversationStorageError("Could not initialize conversation storage") from exc

    def _connect(self):
        connection = self._memory_connection or sqlite3.connect(self._dsn, uri=self._uri, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT,
                    timestamp TEXT NOT NULL,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
                    ON conversation_messages(conversation_id, sequence);
                CREATE TABLE IF NOT EXISTS preferences (
                    preference_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    category TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_preferences_conversation
                    ON preferences(conversation_id, created_at);
            """)

    def _run(self, operation):
        try:
            with self._connect() as db:
                return operation(db)
        except sqlite3.Error as exc:
            raise ConversationStorageError("Conversation storage operation failed") from exc

    def create(self, conversation):
        def op(db):
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (
                conversation.conversation_id, conversation.created_at.isoformat(),
                conversation.updated_at.isoformat(), conversation.status,
            ))
            return conversation
        return self._run(op)

    def get(self, conversation_id):
        def op(db):
            row = db.execute("SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            return Conversation(**dict(row)) if row else None
        return self._run(op)

    def list_conversations(self, limit=100):
        def op(db):
            rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
            return [Conversation(**dict(row)) for row in rows]
        return self._run(op)

    def delete(self, conversation_id):
        return self._run(lambda db: db.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,)).rowcount > 0)

    def append(self, message):
        def op(db):
            db.execute("""INSERT INTO conversation_messages
                (message_id, conversation_id, role, content, timestamp, tool_calls, tool_call_id, name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
                message.message_id, message.conversation_id, message.role, message.content,
                message.timestamp.isoformat(), json.dumps(message.tool_calls) if message.tool_calls is not None else None,
                message.tool_call_id, message.name,
            ))
            db.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                       (message.timestamp.isoformat(), message.conversation_id))
            return message
        return self._run(op)

    def messages(self, conversation_id):
        def op(db):
            rows = db.execute("SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY sequence", (conversation_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item.pop("sequence")
                item["tool_calls"] = json.loads(item["tool_calls"]) if item["tool_calls"] else None
                result.append(ConversationMessage(**item))
            return result
        return self._run(op)

    def save_preference(self, preference):
        def op(db):
            db.execute("INSERT INTO preferences VALUES (?, ?, ?, ?, ?)", (
                preference.preference_id, preference.conversation_id, preference.category,
                preference.value, preference.created_at.isoformat(),
            ))
            return preference
        return self._run(op)

    def preferences(self, conversation_id):
        def op(db):
            rows = db.execute("SELECT * FROM preferences WHERE conversation_id=? ORDER BY created_at", (conversation_id,)).fetchall()
            return [Preference(**dict(row)) for row in rows]
        return self._run(op)

    def delete_preference(self, conversation_id, preference_id):
        return self._run(lambda db: db.execute("DELETE FROM preferences WHERE conversation_id=? AND preference_id=?", (conversation_id, preference_id)).rowcount > 0)

    def clear_preferences(self, conversation_id):
        return self._run(lambda db: db.execute("DELETE FROM preferences WHERE conversation_id=?", (conversation_id,)).rowcount)
