import streamlit_authenticator as stauth

names = ["Parth Sharma"]
usernames = ["parth"]

# hashed password (we’ll generate below)
passwords = ["1234"]

hashed_passwords = stauth.Hasher(passwords).generate()

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
