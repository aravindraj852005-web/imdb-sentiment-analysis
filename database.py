import sqlite3

DB_NAME = "movie_reviews.db"


def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER,
            user_name TEXT,
            rating INTEGER,
            emoji TEXT,
            comment TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_review(movie_id, user_name, rating, emoji, comment):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reviews
        (movie_id, user_name, rating, emoji, comment)
        VALUES (?, ?, ?, ?, ?)
    """, (movie_id, user_name, rating, emoji, comment))

    conn.commit()
    conn.close()


def get_reviews(movie_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT review_id, movie_id, user_name, rating, emoji, comment
        FROM reviews
        WHERE movie_id = ?
    """, (movie_id,))

    data = cursor.fetchall()

    conn.close()
    return data