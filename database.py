
import sqlite3
import json

DB_NAME = "chat_history.db"


def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                messages TEXT NOT NULL,
                final_output TEXT NOT NULL,
                results TEXT
            )
        """)
        conn.commit()


def save_chat(task, messages, final_output, results=None):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chats
            (task, messages, final_output, results)
            VALUES (?, ?, ?, ?)
            """,
            (
                task,
                json.dumps(messages),
                final_output,
                json.dumps(results) if results is not None else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def load_chats():
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute(
            """
            SELECT id, task, messages, final_output, results
            FROM chats
            ORDER BY id ASC
            """
        ).fetchall()

    return [
        {
            "id": row[0],
            "task": row[1],
            "messages": json.loads(row[2]),
            "final_output": row[3],
            "results": json.loads(row[4]) if row[4] else None,
        }
        for row in rows
    ]


def update_chat(chat_id, messages, final_output):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            UPDATE chats
            SET messages = ?, final_output = ?
            WHERE id = ?
            """,
            (json.dumps(messages), final_output, chat_id),
        )
        conn.commit()


def delete_chat(chat_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            "DELETE FROM chats WHERE id = ?",
            (chat_id,),
        )
        conn.commit()
