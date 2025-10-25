from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, text, DECIMAL, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import json
import os
import time


DATABASE_URL = "postgresql+psycopg2://nikitaurovsky:password123@localhost:5432/hack_chapmani"


# Создаем движок базы данных с дополнительными настройками
engine = create_engine(
    DATABASE_URL,
    # Настройки пула соединений
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Проверка соединения перед использованием
    pool_recycle=3600,   # Переподключение каждые 3600 секунд
    
    # Настройки для удаленных баз данных
    connect_args={
        "connect_timeout": 10,  # Таймаут подключения 10 секунд
        "application_name": "GeoDataAPI",  # Имя приложения в логах БД
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Глобальная сессия для совместимости с DB.py
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# Модели базы данных
# Модели из DB.py
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

    locations = relationship("Locations", back_populates="user")

class Locations(Base):
    __tablename__ = "locations"

    location_id = Column(Integer, primary_key=True, autoincrement=True)  # Автогенерируемый ID
    original_id = Column(Integer, nullable=False)  # Исходный ID из файла
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    latitude = Column(DECIMAL(10, 8), nullable=False)
    longitude = Column(DECIMAL(11, 8), nullable=False)
    
    # Новые поля из Excel файла
    address = Column(String, nullable=False)  # Адрес объекта
    work_start = Column(String, nullable=False)  # Время начала рабочего дня
    work_end = Column(String, nullable=False)  # Время окончания рабочего дня
    lunch_start = Column(String, nullable=False)  # Время начала обеда
    lunch_end = Column(String, nullable=False)  # Время окончания обеда
    client_level = Column(String, nullable=False, default="Standart")  # Уровень клиента (VIP/Standart)

    user = relationship("User", back_populates="locations")

# Дополнительные модели для API
class Address(Base):
    __tablename__ = "addresses"
    
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, nullable=False)
    address1 = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    client_level = Column(String, default="Standart")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ClientVisit(Base):
    __tablename__ = "client_visits"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, nullable=False)
    visited = Column(Boolean, default=False)
    visit_time = Column(DateTime)
    notes = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True)
    original_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    records_count = Column(Integer, default=0)
    status = Column(String, default="processed")

class Route(Base):
    __tablename__ = "routes"
    
    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(String, unique=True, index=True)
    route_type = Column(String, nullable=False)  # "ai" or "base"
    total_distance = Column(Float)
    total_time = Column(Float)
    waypoints = Column(JSON)  # JSON field
    created_at = Column(DateTime, default=datetime.utcnow)


# Функция для тестирования подключения
def test_connection():
    """Тестирует подключение к базе данных"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("[OK] Подключение к базе данных успешно")
            return True
    except Exception as e:
        print(f"[ERROR] Ошибка подключения к базе данных: {e}")
        return False

# Создаем таблицы
def create_tables():
    """Создает все таблицы в базе данных"""
    try:
        Base.metadata.create_all(bind=engine)
        print("[OK] Таблицы созданы/проверены")
    except Exception as e:
        print(f"[ERROR] Ошибка создания таблиц: {e}")
        raise

# Зависимость для получения сессии базы данных
def get_db():
    """Генератор для получения сессии базы данных"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"[ERROR] Ошибка сессии базы данных: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Функции для работы с базой данных
class DatabaseManager:
    def __init__(self, db: Session):
        self.db = db
    
    # Работа с адресами
    def get_addresses(self, skip: int = 0, limit: int = 100, client_level: str = None):
        query = self.db.query(Address)
        if client_level:
            query = query.filter(Address.client_level == client_level)
        return query.offset(skip).limit(limit).all()
    
    def get_address_by_id(self, address_id: int):
        return self.db.query(Address).filter(Address.id == address_id).first()
    
    def create_address(self, address_data: dict):
        db_address = Address(**address_data)
        self.db.add(db_address)
        self.db.commit()
        self.db.refresh(db_address)
        return db_address
    
    def update_address(self, address_id: int, address_data: dict):
        db_address = self.db.query(Address).filter(Address.id == address_id).first()
        if db_address:
            for key, value in address_data.items():
                setattr(db_address, key, value)
            db_address.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(db_address)
        return db_address
    
    def delete_address(self, address_id: int):
        db_address = self.db.query(Address).filter(Address.id == address_id).first()
        if db_address:
            self.db.delete(db_address)
            self.db.commit()
        return db_address
    
    def get_addresses_count(self):
        return self.db.query(Address).count()
    
    def get_addresses_stats(self):
        # Статистика по уровням клиентов
        from sqlalchemy import func
        stats = self.db.query(
            Address.client_level,
            func.count(Address.id).label('count')
        ).group_by(Address.client_level).all()
        
        client_levels = {stat.client_level: stat.count for stat in stats}
        
        # Географические границы
        bounds = self.db.query(
            func.min(Address.lat).label('min_lat'),
            func.max(Address.lat).label('max_lat'),
            func.min(Address.lon).label('min_lon'),
            func.max(Address.lon).label('max_lon'),
            func.avg(Address.lat).label('center_lat'),
            func.avg(Address.lon).label('center_lon')
        ).first()
        
        return {
            "total_addresses": self.get_addresses_count(),
            "client_levels": client_levels,
            "geographic_bounds": {
                "min_lat": bounds.min_lat,
                "max_lat": bounds.max_lat,
                "min_lon": bounds.min_lon,
                "max_lon": bounds.max_lon
            },
            "center": {
                "lat": bounds.center_lat,
                "lon": bounds.center_lon
            }
        }
    
    # Работа с посещениями
    def create_visit(self, visit_data: dict):
        db_visit = ClientVisit(**visit_data)
        self.db.add(db_visit)
        self.db.commit()
        self.db.refresh(db_visit)
        return db_visit
    
    def get_visits(self, date_from=None, date_to=None, client_levels=None):
        query = self.db.query(ClientVisit)
        if date_from:
            query = query.filter(ClientVisit.timestamp >= date_from)
        if date_to:
            query = query.filter(ClientVisit.timestamp <= date_to)
        return query.all()
    
    def get_visits_with_addresses(self, date_from=None, date_to=None, client_levels=None):
        from sqlalchemy.orm import joinedload
        query = self.db.query(ClientVisit)
        if date_from:
            query = query.filter(ClientVisit.timestamp >= date_from)
        if date_to:
            query = query.filter(ClientVisit.timestamp <= date_to)
        
        visits = query.all()
        result = []
        
        for visit in visits:
            address = self.get_address_by_id(visit.client_id)
            if address:
                result.append({
                    "client_id": visit.client_id,
                    "client_address": address.address1,
                    "client_level": address.client_level,
                    "visited": visit.visited,
                    "visit_time": visit.visit_time.isoformat() if visit.visit_time else None,
                    "notes": visit.notes,
                    "timestamp": visit.timestamp.isoformat()
                })
        
        return result
    
    # Работа с файлами
    def create_uploaded_file(self, file_data: dict):
        db_file = UploadedFile(**file_data)
        self.db.add(db_file)
        self.db.commit()
        self.db.refresh(db_file)
        return db_file
    
    def get_uploaded_files(self):
        return self.db.query(UploadedFile).all()
    
    # Работа с маршрутами
    def create_route(self, route_data: dict):
        db_route = Route(**route_data)
        self.db.add(db_route)
        self.db.commit()
        self.db.refresh(db_route)
        return db_route
    
    def get_route_by_id(self, route_id: str):
        return self.db.query(Route).filter(Route.route_id == route_id).first()
    
    # Работа с локациями (из DB.py)
    def create_location(self, location_data: dict):
        """Создает новую локацию"""
        db_location = Locations(**location_data)
        self.db.add(db_location)
        self.db.commit()
        self.db.refresh(db_location)
        return db_location
    
    def get_location_by_id(self, location_id: int):
        """Получает локацию по ID"""
        return self.db.query(Locations).filter(Locations.location_id == location_id).first()
    
    def get_locations_by_user(self, user_id: int):
        """Получает все локации пользователя"""
        return self.db.query(Locations).filter(Locations.user_id == user_id).all()
    
    def update_location_status(self, location_id: int, is_active: bool):
        """Обновляет статус локации"""
        location = self.db.query(Locations).filter(Locations.location_id == location_id).first()
        if location:
            location.is_active = is_active
            self.db.commit()
            self.db.refresh(location)
        return location
    
    def get_all_locations(self):
        """Получает все локации"""
        return self.db.query(Locations).all()
    
    def delete_location(self, location_id: int):
        """Удаляет локацию"""
        location = self.db.query(Locations).filter(Locations.location_id == location_id).first()
        if location:
            self.db.delete(location)
            self.db.commit()
        return location
    
    # Работа с пользователями (из DB.py)
    def create_user(self, user_data: dict):
        """Создает нового пользователя"""
        db_user = User(**user_data)
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def get_user_by_id(self, user_id: int):
        """Получает пользователя по ID"""
        return self.db.query(User).filter(User.user_id == user_id).first()
    
    def get_all_users(self):
        """Получает всех пользователей"""
        return self.db.query(User).all()
    
    def update_user(self, user_id: int, user_data: dict):
        """Обновляет данные пользователя"""
        user = self.get_user_by_id(user_id)
        if user:
            for key, value in user_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            self.db.commit()
            self.db.refresh(user)
        return user
    
    def delete_user(self, user_id: int):
        """Удаляет пользователя"""
        user = self.get_user_by_id(user_id)
        if user:
            self.db.delete(user)
            self.db.commit()
        return user

# Примеры различных URL для подключения к удаленным базам данных
def get_database_urls_examples():
    """Возвращает примеры URL для различных типов подключений"""
    return {
        "local_postgresql": "postgresql+psycopg2://username:password@localhost:5432/database_name",
        "remote_postgresql": "postgresql+psycopg2://username:password@remote_host:5432/database_name",
        "postgresql_with_ssl": "postgresql+psycopg2://username:password@host:5432/database?sslmode=require",
        "heroku_postgres": "postgresql+psycopg2://user:pass@host:5432/dbname?sslmode=require",
        "railway_postgres": "postgresql+psycopg2://postgres:password@containers-us-west-1.railway.app:5432/railway",
        "supabase_postgres": "postgresql+psycopg2://postgres:password@db.project.supabase.co:5432/postgres",
        "aws_rds": "postgresql+psycopg2://username:password@your-instance.region.rds.amazonaws.com:5432/database_name",
        "google_cloud_sql": "postgresql+psycopg2://username:password@/database_name?host=/cloudsql/project:region:instance",
        "azure_postgres": "postgresql+psycopg2://username@server:password@server.postgres.database.azure.com:5432/database_name?sslmode=require"
    }

# Функции для работы с сессией (из DB.py)
def create_tables_legacy():
    """Создание таблиц с использованием глобальной сессии (совместимость с DB.py)"""
    Base.metadata.create_all(engine)
    print("Таблицы созданы успешно!")

def get_session():
    """Получение новой сессии"""
    return Session()

def close_session(session_to_close=None):
    """Закрытие сессии"""
    if session_to_close:
        session_to_close.close()
    else:
        session.close()

# Функции для работы с данными без закрытия сессии (из DB.py)
def get_all_locations():
    """Получение всех локаций"""
    try:
        locations = session.query(Locations).all()
        return locations
    except Exception as e:
        print(f"Ошибка при получении локаций: {e}")
        return []

def get_locations_by_user(user_id: int):
    """Получение локаций пользователя"""
    try:
        locations = session.query(Locations).filter(Locations.user_id == user_id).all()
        return locations
    except Exception as e:
        print(f"Ошибка при получении локаций пользователя: {e}")
        return []

def update_location_status(location_id: int, is_active: bool):
    """Обновление статуса локации"""
    try:
        location = session.query(Locations).filter(Locations.location_id == location_id).first()
        
        if location:
            location.is_active = is_active
            session.commit()
            print(f"Статус локации {location_id} обновлен на {is_active}")
            return True
        else:
            print(f"Локация с ID {location_id} не найдена")
            return False
    except Exception as e:
        session.rollback()
        print(f"Ошибка при обновлении статуса локации: {e}")
        return False

if __name__ == "__main__":
    # Тестируем подключение
    print("Тестирование подключения к базе данных...")
    if test_connection():
        print("Создание таблиц...")
        create_tables()
        print("Готово!")
    else:
        print("Не удалось подключиться к базе данных")
