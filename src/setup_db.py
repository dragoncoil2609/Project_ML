import sqlite3

print("Đang tạo database...")
try:
    # Kết nối (sẽ tự tạo file 'users.db' nếu chưa có)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # 1. Tạo bảng 'users'
    # (username là duy nhất, password đã được mã hóa)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        name TEXT
    )
    ''')

    # 2. Tạo bảng 'history'
    # (username là 'khóa ngoại' để liên kết với bảng 'users')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        analysis_type TEXT NOT NULL, 
        input_text TEXT,
        result_label TEXT,
        result_score REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')

    conn.commit()
    conn.close()
    print("Tạo database 'users.db' và các bảng (users, history) thành công!")

except Exception as e:
    print(f"Lỗi khi tạo DB: {e}")