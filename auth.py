import streamlit_authenticator as stauth

names = ["Parth Sharma"]
usernames = ["parth"]

# IMPORTANT: pre-hashed password (not plain text)
# password = "1234"
hashed_passwords = stauth.Hasher(["1234"]).generate()

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
