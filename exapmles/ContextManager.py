from contextlib import contextmanager
from requests import session

URL = 'http://myserver.com'


@contextmanager
def context(s):
    try:
        # получаем токен
        token = s.post(f"{URL}/auth",
                       json=dict(user="user",
                                 password="password")).json().get('token')
        # добавляем во все хедеры сессии токен для авторизации
        s.headers.update({'Authorization': token})
        yield s
    finally:
        s.close()


with context(session()) as s:
    s.get(f"{URL}/api")
    # ...

