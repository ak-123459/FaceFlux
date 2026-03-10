import pymysql

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "frs_user",
    "password": "StrongPassword",
    "database": "attendance_db"
}



def test_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)

        print("✅ Connection successful!")

        with conn.cursor() as cursor:
            cursor.execute("SELECT DATABASE();")
            db = cursor.fetchone()

            print("Connected to database:", db[0])

        conn.close()

    except Exception as e:
        print("❌ Connection failed")
        print("Error:", e)


if __name__ == "__main__":
    test_connection()