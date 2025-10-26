#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединенный сервер FastAPI с подключением к удаленной БД и всеми функциями
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
import json
import os
import csv
import uuid
from datetime import datetime
import shutil
from sqlalchemy.orm import Session

# Импортируем настройки удаленной базы данных
from database_remote import get_db, DatabaseManager, create_tables, test_connection

# Импортируем функции из button_for_front.py
from button_for_front import (
    upload_client_file, 
    convert_to_json, 
    process_json_to_locations,
    process_client_file_complete,
    register_user,
    find_user_by_phone,
    get_user_available_locations,
    mark_location_visited,
    set_stay_period,
    check_stay_period
)

# Функция для управления жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[INFO] Проверка подключения к базе данных...")
    if test_connection():
        print("[INFO] Создание/проверка таблиц...")
        create_tables()
        print("[OK] База данных готова к работе")
    else:
        print("[WARNING] Не удалось подключиться к базе данных. API будет работать в ограниченном режиме.")
    yield
    # Shutdown (если нужно что-то закрыть)

# Создаем экземпляр FastAPI
app = FastAPI(
    title="GeoData API - Unified",
    description="Объединенный REST API для работы с геоданными адресов с поддержкой удаленной PostgreSQL",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Модели данных (Pydantic)
class AddressBase(BaseModel):
    address: str
    address1: str
    lat: float
    lon: float
    client_level: str = "Standart"

class AddressCreate(AddressBase):
    pass

class AddressUpdate(BaseModel):
    address: Optional[str] = None
    address1: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    client_level: Optional[str] = None

class Address(AddressBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Новые модели для дополнительных эндпоинтов
class RouteRequest(BaseModel):
    client_ids: List[int]
    start_lat: float
    start_lon: float
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    algorithm: str = "ai"  # "ai" или "base"

class RouteResponse(BaseModel):
    route_id: str
    total_distance: float
    total_time: float
    waypoints: List[dict]
    optimized: bool

class ClientStatusUpdate(BaseModel):
    client_id: int
    visited: bool
    visit_time: Optional[str] = None
    notes: Optional[str] = None

class ExportRequest(BaseModel):
    format: str = "json"  # "json", "csv", "excel"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    client_levels: Optional[List[str]] = None

# Новые модели для работы с button_for_front.py
class UserRegistration(BaseModel):
    first_name: str
    last_name: str
    phone_number: str
    password: str

class UserLogin(BaseModel):
    phone_number: str
    password: str

class LocationVisit(BaseModel):
    user_id: int
    location_id: int

class StayPeriodRequest(BaseModel):
    user_id: int
    days: int


# Базовые эндпоинты
@app.get("/")
async def root():
    """Главная страница фронтенда"""
    return FileResponse("frontend/index.html")

@app.get("/api")
async def api_root():
    """API корневой эндпоинт"""
    return {
        "message": "Добро пожаловать в GeoData API - Объединенная версия!",
        "version": "3.0.0",
        "docs": "/docs",
        "database": "Remote PostgreSQL",
        "features": [
            "CRUD операции с адресами",
            "Поиск ближайших адресов",
            "AI и базовые маршруты",
            "Загрузка файлов",
            "Статистика и отчеты",
            "Управление посещениями"
        ]
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Проверка состояния API и базы данных"""
    try:
        db_manager = DatabaseManager(db)
        addresses_count = db_manager.get_addresses_count()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "addresses_count": addresses_count,
            "database": "connected",
            "database_type": "PostgreSQL",
            "version": "3.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "database": "disconnected"
        }

# CRUD операции для адресов с PostgreSQL
@app.get("/addresses", response_model=List[Address])
async def get_addresses(
    skip: int = 0,
    limit: int = 100,
    client_level: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Получить список всех адресов с возможностью фильтрации"""
    db_manager = DatabaseManager(db)
    addresses = db_manager.get_addresses(skip=skip, limit=limit, client_level=client_level)
    return addresses

@app.get("/addresses/stats")
async def get_address_stats(db: Session = Depends(get_db)):
    """Получить статистику по адресам"""
    db_manager = DatabaseManager(db)
    stats = db_manager.get_addresses_stats()
    return stats

@app.get("/addresses/{address_id}", response_model=Address)
async def get_address(address_id: int, db: Session = Depends(get_db)):
    """Получить конкретный адрес по ID"""
    db_manager = DatabaseManager(db)
    address = db_manager.get_address_by_id(address_id)
    if not address:
        raise HTTPException(status_code=404, detail="Адрес не найден")
    return address

@app.post("/addresses", response_model=Address)
async def create_address(address: AddressCreate, db: Session = Depends(get_db)):
    """Создать новый адрес"""
    db_manager = DatabaseManager(db)
    address_data = address.model_dump()
    new_address = db_manager.create_address(address_data)
    return new_address

@app.put("/addresses/{address_id}", response_model=Address)
async def update_address(
    address_id: int, 
    address_update: AddressUpdate, 
    db: Session = Depends(get_db)
):
    """Обновить адрес"""
    db_manager = DatabaseManager(db)
    update_data = address_update.model_dump(exclude_unset=True)
    updated_address = db_manager.update_address(address_id, update_data)
    if not updated_address:
        raise HTTPException(status_code=404, detail="Адрес не найден")
    return updated_address

@app.delete("/addresses/{address_id}")
async def delete_address(address_id: int, db: Session = Depends(get_db)):
    """Удалить адрес"""
    db_manager = DatabaseManager(db)
    deleted_address = db_manager.delete_address(address_id)
    if not deleted_address:
        raise HTTPException(status_code=404, detail="Адрес не найден")
    return {"message": "Адрес удален", "deleted_address": deleted_address}

# Специальные эндпоинты для геоданных
@app.get("/addresses/nearby")
async def get_nearby_addresses(
    lat: float,
    lon: float,
    radius: float = 1.0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Получить адреса в радиусе от указанных координат"""
    import math
    
    def calculate_distance(lat1, lon1, lat2, lon2):
        """Вычисляет расстояние между двумя точками в километрах"""
        R = 6371  # Радиус Земли в км
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    db_manager = DatabaseManager(db)
    all_addresses = db_manager.get_addresses(limit=1000)  # Получаем больше адресов для поиска
    
    nearby_addresses = []
    for address in all_addresses:
        distance = calculate_distance(lat, lon, address.lat, address.lon)
        if distance <= radius:
            address_dict = {
                "id": address.id,
                "address": address.address,
                "address1": address.address1,
                "lat": address.lat,
                "lon": address.lon,
                "client_level": address.client_level,
                "distance_km": round(distance, 3)
            }
            nearby_addresses.append(address_dict)
    
    # Сортируем по расстоянию
    nearby_addresses.sort(key=lambda x: x["distance_km"])
    
    return nearby_addresses[:limit]

# Эндпоинт для массовой загрузки данных
@app.post("/addresses/bulk")
async def create_addresses_bulk(addresses: List[AddressCreate], db: Session = Depends(get_db)):
    """Создать несколько адресов одновременно"""
    db_manager = DatabaseManager(db)
    new_addresses = []
    
    for address in addresses:
        address_data = address.model_dump()
        new_address = db_manager.create_address(address_data)
        new_addresses.append(new_address)
    
    return {
        "message": f"Создано {len(new_addresses)} адресов",
        "created_addresses": new_addresses
    }

# ===== ЭНДПОИНТЫ ДЛЯ ЗАГРУЗКИ ФАЙЛОВ =====

@app.post("/api/upload")
async def upload_clients_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Загрузка файла клиентов"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    # Проверяем расширение файла
    if not file.filename.endswith(('.csv', '.json', '.xlsx')):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы CSV, JSON и XLSX")
    
    # Создаем папку для загруженных файлов
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Генерируем уникальное имя файла
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(upload_dir, f"{file_id}{file_extension}")
    
    try:
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Обрабатываем файл в зависимости от типа
        processed_data = []
        if file.filename.endswith('.csv'):
            processed_data = process_csv_file(file_path)
        elif file.filename.endswith('.json'):
            processed_data = process_json_file(file_path)
        
        # Сохраняем адреса в базу данных
        db_manager = DatabaseManager(db)
        created_addresses = []
        for address_data in processed_data:
            try:
                new_address = db_manager.create_address(address_data)
                created_addresses.append(new_address)
            except Exception as e:
                print(f"Ошибка создания адреса: {e}")
                continue
        
        # Сохраняем информацию о файле в базу данных
        file_info = {
            "file_id": file_id,
            "original_name": file.filename,
            "file_path": file_path,
            "upload_time": datetime.now(),
            "records_count": len(created_addresses),
            "status": "processed"
        }
        try:
            db_manager.create_uploaded_file(file_info)
        except Exception as e:
            print(f"Ошибка сохранения информации о файле: {e}")
        
        return {
            "message": "Файл успешно загружен",
            "file_id": file_id,
            "records_processed": len(created_addresses),
            "created_addresses": created_addresses,
            "file_info": file_info
        }
        
    except Exception as e:
        # Удаляем файл в случае ошибки
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

def process_csv_file(file_path):
    """Обрабатывает CSV файл с клиентами"""
    processed = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Нормализуем данные из CSV
                if 'address' in row and 'lat' in row and 'lon' in row:
                    processed.append({
                        'address': row['address'],
                        'address1': row.get('address1', ''),
                        'lat': float(row['lat']),
                        'lon': float(row['lon']),
                        'client_level': row.get('client_level', 'Standart')
                    })
    except Exception as e:
        raise Exception(f"Ошибка обработки CSV: {e}")
    return processed

def process_json_file(file_path):
    """Обрабатывает JSON файл с клиентами"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except Exception as e:
        raise Exception(f"Ошибка обработки JSON: {e}")

# ===== ЭНДПОИНТЫ ДЛЯ МАРШРУТИЗАЦИИ =====

@app.post("/api/route", response_model=RouteResponse)
async def get_ai_route(route_request: RouteRequest, db: Session = Depends(get_db)):
    """Получение оптимального (AI) маршрута"""
    if route_request.algorithm != "ai":
        raise HTTPException(status_code=400, detail="Этот эндпоинт только для AI маршрутов")
    
    # Получаем адреса клиентов из базы данных
    db_manager = DatabaseManager(db)
    client_addresses = []
    for client_id in route_request.client_ids:
        client = db_manager.get_address_by_id(client_id)
        if not client:
            raise HTTPException(status_code=404, detail=f"Клиент с ID {client_id} не найден")
        client_addresses.append(client)
    
    # Генерируем AI маршрут
    route_id = str(uuid.uuid4())
    waypoints = generate_ai_route(
        start_lat=route_request.start_lat,
        start_lon=route_request.start_lon,
        client_addresses=client_addresses,
        end_lat=route_request.end_lat,
        end_lon=route_request.end_lon
    )
    
    # Рассчитываем общее расстояние и время
    total_distance = calculate_total_distance(waypoints)
    total_time = calculate_total_time(total_distance)
    
    # Сохраняем маршрут в базу данных
    route_data = {
        "route_id": route_id,
        "route_type": "ai",
        "total_distance": total_distance,
        "total_time": total_time,
        "waypoints": waypoints
    }
    db_manager.create_route(route_data)
    
    return RouteResponse(
        route_id=route_id,
        total_distance=total_distance,
        total_time=total_time,
        waypoints=waypoints,
        optimized=True
    )

@app.post("/api/route/base", response_model=RouteResponse)
async def get_base_route(route_request: RouteRequest, db: Session = Depends(get_db)):
    """Получение базового маршрута"""
    if route_request.algorithm != "base":
        raise HTTPException(status_code=400, detail="Этот эндпоинт только для базовых маршрутов")
    
    # Получаем адреса клиентов из базы данных
    db_manager = DatabaseManager(db)
    client_addresses = []
    for client_id in route_request.client_ids:
        client = db_manager.get_address_by_id(client_id)
        if not client:
            raise HTTPException(status_code=404, detail=f"Клиент с ID {client_id} не найден")
        client_addresses.append(client)
    
    # Генерируем базовый маршрут
    route_id = str(uuid.uuid4())
    waypoints = generate_base_route(
        start_lat=route_request.start_lat,
        start_lon=route_request.start_lon,
        client_addresses=client_addresses,
        end_lat=route_request.end_lat,
        end_lon=route_request.end_lon
    )
    
    # Рассчитываем общее расстояние и время
    total_distance = calculate_total_distance(waypoints)
    total_time = calculate_total_time(total_distance)
    
    # Сохраняем маршрут в базу данных
    route_data = {
        "route_id": route_id,
        "route_type": "base",
        "total_distance": total_distance,
        "total_time": total_time,
        "waypoints": waypoints
    }
    db_manager.create_route(route_data)
    
    return RouteResponse(
        route_id=route_id,
        total_distance=total_distance,
        total_time=total_time,
        waypoints=waypoints,
        optimized=False
    )

def generate_ai_route(start_lat, start_lon, client_addresses, end_lat=None, end_lon=None):
    """Генерирует AI-оптимизированный маршрут"""
    import math
    
    waypoints = []
    
    # Начальная точка
    waypoints.append({
        "order": 0,
        "type": "start",
        "lat": start_lat,
        "lon": start_lon,
        "address": "Начальная точка",
        "client_id": None
    })
    
    # Сортируем клиентов по расстоянию от начальной точки (упрощенный AI алгоритм)
    sorted_clients = sorted(client_addresses, key=lambda x: 
        math.sqrt((x.lat - start_lat)**2 + (x.lon - start_lon)**2))
    
    # Добавляем клиентов в маршрут
    for i, client in enumerate(sorted_clients):
        waypoints.append({
            "order": i + 1,
            "type": "client",
            "lat": client.lat,
            "lon": client.lon,
            "address": client.address1,
            "client_id": client.id,
            "client_level": client.client_level
        })
    
    # Конечная точка
    if end_lat and end_lon:
        waypoints.append({
            "order": len(waypoints),
            "type": "end",
            "lat": end_lat,
            "lon": end_lon,
            "address": "Конечная точка",
            "client_id": None
        })
    
    return waypoints

def generate_base_route(start_lat, start_lon, client_addresses, end_lat=None, end_lon=None):
    """Генерирует базовый маршрут (простой порядок)"""
    waypoints = []
    
    # Начальная точка
    waypoints.append({
        "order": 0,
        "type": "start",
        "lat": start_lat,
        "lon": start_lon,
        "address": "Начальная точка",
        "client_id": None
    })
    
    # Добавляем клиентов в том порядке, как они пришли
    for i, client in enumerate(client_addresses):
        waypoints.append({
            "order": i + 1,
            "type": "client",
            "lat": client.lat,
            "lon": client.lon,
            "address": client.address1,
            "client_id": client.id,
            "client_level": client.client_level
        })
    
    # Конечная точка
    if end_lat and end_lon:
        waypoints.append({
            "order": len(waypoints),
            "type": "end",
            "lat": end_lat,
            "lon": end_lon,
            "address": "Конечная точка",
            "client_id": None
        })
    
    return waypoints

def calculate_total_distance(waypoints):
    """Рассчитывает общее расстояние маршрута в км"""
    import math
    
    total_distance = 0
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]["lat"], waypoints[i]["lon"]
        lat2, lon2 = waypoints[i + 1]["lat"], waypoints[i + 1]["lon"]
        
        # Формула гаверсинуса для расчета расстояния
        R = 6371  # Радиус Земли в км
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        total_distance += distance
    
    return round(total_distance, 2)

def calculate_total_time(distance_km):
    """Рассчитывает примерное время в пути в часах"""
    # Предполагаем среднюю скорость 30 км/ч в городе
    return round(distance_km / 30, 2)

# ===== ЭНДПОИНТЫ ДЛЯ УПРАВЛЕНИЯ ПОСЕЩЕНИЯМИ =====

@app.post("/api/client/status")
async def update_client_status(status_update: ClientStatusUpdate, db: Session = Depends(get_db)):
    """Отметка посещения клиента"""
    # Проверяем, существует ли клиент
    db_manager = DatabaseManager(db)
    client = db_manager.get_address_by_id(status_update.client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Клиент с ID {status_update.client_id} не найден")
    
    # Создаем запись о посещении
    visit_data = {
        "client_id": status_update.client_id,
        "visited": status_update.visited,
        "visit_time": datetime.fromisoformat(status_update.visit_time) if status_update.visit_time else datetime.now(),
        "notes": status_update.notes
    }
    
    visit_record = db_manager.create_visit(visit_data)
    
    return {
        "message": "Статус клиента обновлен",
        "visit_record": {
            "client_id": visit_record.client_id,
            "visited": visit_record.visited,
            "visit_time": visit_record.visit_time.isoformat() if visit_record.visit_time else None,
            "notes": visit_record.notes,
            "timestamp": visit_record.timestamp.isoformat()
        }
    }

# ===== ЭНДПОИНТЫ ДЛЯ ЭКСПОРТА =====

@app.get("/api/export")
async def export_report(
    format: str = "json",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    client_levels: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Выгрузка итогового отчёта"""
    db_manager = DatabaseManager(db)
    
    # Получаем данные о посещениях с адресами
    export_data = db_manager.get_visits_with_addresses(
        date_from=date_from,
        date_to=date_to,
        client_levels=client_levels
    )
    
    if format.lower() == "csv":
        # Создаем CSV файл
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = f"exports/{filename}"
        os.makedirs("exports", exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if export_data:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='text/csv'
        )
    
    elif format.lower() == "json":
        # Возвращаем JSON
        return {
            "export_info": {
                "format": "json",
                "total_records": len(export_data),
                "date_from": date_from,
                "date_to": date_to,
                "client_levels": client_levels,
                "export_time": datetime.now().isoformat()
            },
            "data": export_data
        }
    
    else:
        raise HTTPException(status_code=400, detail="Поддерживаются только форматы: json, csv")

# ===== НОВЫЕ ЭНДПОИНТЫ ДЛЯ РАБОТЫ С BUTTON_FOR_FRONT.PY =====

@app.post("/api/upload-advanced")
async def upload_advanced_file(file: UploadFile = File(...)):
    """Расширенная загрузка файлов с поддержкой Excel и полной обработкой"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    # Проверяем расширение файла
    if not file.filename.endswith(('.csv', '.json', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы CSV, JSON и Excel (.xlsx, .xls)")
    
    # Создаем папку для загруженных файлов
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Генерируем уникальное имя файла
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(upload_dir, f"{file_id}{file_extension}")
    
    try:
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Используем функцию из button_for_front.py для полной обработки
        result = process_client_file_complete(file_path, user_id=1)
        
        # Удаляем временный файл
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if result["success"]:
            return {
                "message": "Файл успешно обработан и загружен в базу данных",
                "file_id": file_id,
                "total_processed": result["total_processed"],
                "total_errors": result["total_errors"],
                "errors": result.get("database_info", {}).get("errors", []),
                "processing_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {result['error']}")
            
    except Exception as e:
        # Удаляем файл в случае ошибки
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

@app.post("/api/register")
async def register_new_user(user_data: UserRegistration):
    """Регистрация нового пользователя"""
    try:
        result = register_user(
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone_number=user_data.phone_number,
            password=user_data.password
        )
        
        if result["success"]:
            return {
                "message": result["message"],
                "user_id": result["user_id"],
                "first_name": result["first_name"],
                "last_name": result["last_name"],
                "registration_time": result["registration_time"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка регистрации: {str(e)}")

@app.post("/api/login")
async def login_user(login_data: UserLogin):
    """Авторизация пользователя"""
    try:
        # Находим пользователя по номеру телефона
        user = find_user_by_phone(login_data.phone_number)
        
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        # Проверяем пароль (простая проверка хеша)
        import hashlib
        password_hash = hashlib.sha256(login_data.password.encode('utf-8')).hexdigest()
        
        if user["password_hash"] != password_hash:
            raise HTTPException(status_code=401, detail="Неверный пароль")
        
        return {
            "message": "Успешная авторизация",
            "user_id": user["user_id"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "phone_number": user["phone_number"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка авторизации: {str(e)}")

@app.get("/api/user/{user_id}/locations")
async def get_user_locations(user_id: int):
    """Получение локаций пользователя"""
    try:
        result = get_user_available_locations(user_id)
        
        if result["success"]:
            return {
                "user_id": result["user_id"],
                "total_locations": result["total_locations"],
                "visited_locations": result["visited_locations"],
                "available_locations": result["available_locations"],
                "locations": result["locations"],
                "query_time": result["query_time"]
            }
        else:
            raise HTTPException(status_code=404, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения локаций: {str(e)}")

@app.post("/api/location/visit")
async def mark_location_as_visited(visit_data: LocationVisit):
    """Отметка посещения локации"""
    try:
        result = mark_location_visited(visit_data.user_id, visit_data.location_id)
        
        if result["success"]:
            return {
                "message": result["message"],
                "user_id": result["user_id"],
                "location_id": result["location_id"],
                "address": result["address"],
                "is_active": result["is_active"],
                "action_time": result["action_time"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка отметки посещения: {str(e)}")

@app.post("/api/stay-period")
async def set_user_stay_period(stay_data: StayPeriodRequest):
    """Установка периода пребывания пользователя"""
    try:
        result = set_stay_period(stay_data.user_id, stay_data.days)
        
        if result["success"]:
            return {
                "message": result["message"],
                "user_id": result["user_id"],
                "stay_days": result["stay_days"],
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "expires_in_hours": result["expires_in_hours"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка установки периода: {str(e)}")

@app.get("/api/stay-period/{user_id}")
async def get_user_stay_period(user_id: int):
    """Получение информации о периоде пребывания пользователя"""
    try:
        result = check_stay_period(user_id)
        
        if result["success"]:
            return {
                "user_id": result["user_id"],
                "current_day": result["current_day"],
                "total_days": result["total_days"],
                "days_remaining": result["days_remaining"],
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "is_active": result["is_active"]
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "user_id": user_id
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения периода: {str(e)}")

@app.delete("/api/user/{user_id}/locations")
async def delete_user_locations_endpoint(user_id: int):
    """Удаление всех локаций пользователя"""
    try:
        from button_for_front import delete_user_locations
        result = delete_user_locations(user_id)
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "deleted_count": result["deleted_count"],
                "user_id": result["user_id"],
                "deletion_time": result["deletion_time"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка удаления локаций: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Запускаем на порту 8000 (основной порт)
    uvicorn.run(app, host="0.0.0.0", port=8000)
