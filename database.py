from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column,ForeignKey, Integer, String, Boolean, DECIMAL
from sqlalchemy.orm import relationship
from sqlalchemy import select
from dotenv import load_dotenv
import os

load_dotenv()
 
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL) # type: ignore
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


#Модель users
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

#Модель locations
class Locations(Base):
    __tablename__ = "locations"
    
    location_id = Column(Integer,primary_key=True)
    user_id = Column(Integer,ForeignKey("user.users_id"),nullable=False)
    is_active = Column(Boolean,nullable=False)
    latitude = Column(DECIMAL(10,8),nullable=False)
    longitude = Column(DECIMAL(11,8),nullable=False)
    
    user = relationship("Users", back_populates="locations")
    
#создание user
def create_user(first_name: str, last_name: str,phone_number: str,password_hash: str) -> User:
    """Создание нового пользователя"""
    new_user = User(
        first_name = first_name,
        last_name = last_name,
        phone_number=phone_number,
        password_hash=password_hash
    )
        
    session.add(new_user)
    session.commit()
        
    return new_user


"""#получение данных
def get_user_by_id(user_id: int):
    
    return session.query(User).filter(User.user_id == user_id).first()

def get_all_users():
    
    return session.query(User).all()


#обновление данных
def update_user(user_id: int, **kwargs):
    
    user = get_user_by_id(user_id)
    if not user:
        return None
    
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    session.commit()
    return user


#удаление пользователя
def delete_user(user_id: int) -> bool:
    
    user = get_user_by_id(user_id)
    if user:
        session.delete(user)
        session.commit()
        return True
    return False"""
    

