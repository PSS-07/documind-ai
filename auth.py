import streamlit_authenticator as stauth


names = ["Parth Sharma"]
usernames = ["parth"]

# 🔐 Pre-generated bcrypt hash for password: 1234
hashed_passwords = [
    "$2b$12$KIXQk1YqK7zXq8WmYx8G2e6pWv6bQ3m2gqFZrP3w2hR6YxQxYxXyO"
]

credentials = {
    "usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
        }
    }
}


authenticator = stauth.Authenticate(
    credentials,
    "documind_cookie",
    "abcdef",
    cookie_expiry_days=1
)
