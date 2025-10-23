from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy import select
from dotenv import load_dotenv
import os

load_dotenv()
 
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


#Модель users
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    login = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    passwor_hash = Column(String, nullable=False)
    
#создание user
def create_user(login: str, email: str,passwor_hash: str) -> User:
    """Создание нового пользователя"""
    new_user = User(
        login=login,
        email=email,
        passwor_hash=passwor_hash
    )
        
    session.add(new_user)
    session.commit()
        
    return new_user


#получение данных
def get_user_by_id(user_id: int):
    """Получить пользователя по ID"""
    return session.query(User).filter(User.user_id == user_id).first()

def get_user_by_login(login: str):
    """Получить пользователя по логину"""
    return session.query(User).filter(User.login == login).first()

def get_user_by_email(email: str):
    """Получить пользователя по email"""
    return session.query(User).filter(User.email == email).first()

def get_all_users():
    """Получить всех пользователей"""
    return session.query(User).all()


def update_user(user_id: int, **kwargs):
    """Обновление данных пользователя"""
    user = get_user_by_id(user_id)
    if not user:
        return None
    
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    session.commit()
    return user


#обновление данных
def update_user_email(user_id: int, new_email: str):
    """Обновление email пользователя"""
    return update_user(user_id, email=new_email)

def update_user_login(user_id: int, new_login: str):
    """Обновление логина пользователя"""
    return update_user(user_id, login=new_login)


#удаление пользователя
def delete_user(user_id: int) -> bool:
    """Удаление пользователя по ID"""
    user = get_user_by_id(user_id)
    if user:
        session.delete(user)
        session.commit()
        return True
    return False
