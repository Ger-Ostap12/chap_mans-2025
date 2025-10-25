#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для обработки файлов клиентов и загрузки в базу данных
"""

import pandas as pd
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile
import shutil

# Импортируем настройки базы данных
from database_remote import get_db, DatabaseManager, session

def upload_client_file(file_path: str, user_id: int = 1) -> Dict[str, Any]:
    """
    1) Загружает файл Excel/CSV с клиентами
    
    Args:
        file_path: Путь к загруженному файлу
        user_id: ID пользователя, который загрузил файл
    
    Returns:
        Dict с информацией о загруженном файле
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        # Определяем тип файла по расширению
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in ['.csv', '.xlsx', '.xls']:
            raise ValueError("Поддерживаются только файлы CSV и Excel (.xlsx, .xls)")
        
        # Читаем файл в зависимости от типа
        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        else:  # Excel файлы
            df = pd.read_excel(file_path)
        
        # Генерируем уникальный ID для файла
        file_id = str(uuid.uuid4())
        
        # Создаем временный JSON файл
        temp_json_path = f"temp_{file_id}.json"
        
        # Конвертируем DataFrame в JSON
        df.to_json(temp_json_path, orient='records', force_ascii=False, indent=2)
        
        return {
            "success": True,
            "file_id": file_id,
            "original_file": file_path,
            "json_file": temp_json_path,
            "records_count": len(df),
            "columns": list(df.columns),
            "user_id": user_id,
            "upload_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

def convert_to_json(file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    2) Переделывает загруженный файл в JSON формат
    
    Args:
        file_path: Путь к исходному файлу
        output_path: Путь для сохранения JSON (если None, создается автоматически)
    
    Returns:
        Dict с информацией о конвертации
    """
    try:
        # Определяем тип файла
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Читаем файл
        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            # Если уже JSON, просто копируем
            if output_path:
                shutil.copy2(file_path, output_path)
            else:
                output_path = file_path
            return {
                "success": True,
                "json_file": output_path,
                "records_count": len(pd.read_json(file_path)),
                "message": "Файл уже в JSON формате"
            }
        else:
            raise ValueError("Неподдерживаемый формат файла")
        
        # Создаем путь для JSON файла
        if not output_path:
            json_filename = f"converted_{uuid.uuid4().hex}.json"
            output_path = json_filename
        
        # Конвертируем в JSON
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        
        return {
            "success": True,
            "json_file": output_path,
            "records_count": len(df),
            "columns": list(df.columns),
            "conversion_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

def process_json_to_locations(json_file_path: str, user_id: int = 1) -> Dict[str, Any]:
    """
    3) Из JSON файла достает и обрабатывает только нужную информацию
    Сохраняет в БД таблицу locations
    
    Args:
        json_file_path: Путь к JSON файлу
        user_id: ID пользователя
    
    Returns:
        Dict с результатами обработки
    """
    try:
        # Читаем JSON файл
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            raise ValueError("JSON файл должен содержать массив объектов")
        
        # Подключаемся к базе данных
        from database_remote import SessionLocal
        db = SessionLocal()
        db_manager = DatabaseManager(db)
        
        processed_locations = []
        errors = []
        
        try:
            for i, item in enumerate(data):
                try:
                    # Извлекаем нужные поля из JSON (соответствуют колонкам Excel)
                    location_id = item.get('Номер объекта') or item.get('id')
                    address = item.get('Адрес объекта') or item.get('address')
                    lat = item.get('Географическая широта') or item.get('lat')
                    lon = item.get('Географическая долгота') or item.get('lon')
                    work_start = item.get('Время начала рабочего дня') or item.get('work_start')
                    work_end = item.get('Время окончания рабочего дня') or item.get('work_end')
                    lunch_start = item.get('Время начала обеда') or item.get('lunch_start')
                    lunch_end = item.get('Время окончания обеда') or item.get('lunch_end')
                    client_level = item.get('Уровень клиента') or item.get('client_level')
                    
                    # Проверяем обязательные поля
                    if location_id is None:
                        errors.append(f"Запись {i}: отсутствует поле 'Номер объекта' или 'id'")
                        continue
                    
                    if address is None:
                        errors.append(f"Запись {i}: отсутствует поле 'Адрес объекта' или 'address'")
                        continue
                    
                    if lat is None or lon is None:
                        errors.append(f"Запись {i}: отсутствуют координаты 'Географическая широта/долгота' или 'lat/lon'")
                        continue
                    
                    # Преобразуем координаты в float
                    try:
                        latitude = float(lat)
                        longitude = float(lon)
                    except (ValueError, TypeError):
                        errors.append(f"Запись {i}: неверный формат координат")
                        continue
                    
                    # Создаем запись для базы данных
                    location_data = {
                        "original_id": int(location_id),  # Исходный ID из файла
                        "user_id": user_id,
                        "is_active": False,  # По умолчанию false
                        "latitude": latitude,
                        "longitude": longitude,
                        "address": str(address),
                        "work_start": str(work_start) if work_start else "09:00",
                        "work_end": str(work_end) if work_end else "18:00",
                        "lunch_start": str(lunch_start) if lunch_start else "13:00",
                        "lunch_end": str(lunch_end) if lunch_end else "14:00",
                        "client_level": str(client_level) if client_level else "Standart"
                    }
                    
                    # Убираем проверку на существующие location_id
                    # Разные пользователи могут загружать одинаковые данные
                    
                    # Сохраняем в базу данных
                    new_location = db_manager.create_location(location_data)
                    processed_locations.append({
                        "location_id": new_location.location_id,
                        "original_id": new_location.original_id,
                        "user_id": new_location.user_id,
                        "latitude": float(new_location.latitude),
                        "longitude": float(new_location.longitude),
                        "is_active": new_location.is_active,
                        "address": new_location.address,
                        "work_start": new_location.work_start,
                        "work_end": new_location.work_end,
                        "lunch_start": new_location.lunch_start,
                        "lunch_end": new_location.lunch_end,
                        "client_level": new_location.client_level
                    })
                    
                except Exception as e:
                    errors.append(f"Запись {i}: ошибка обработки - {str(e)}")
                    continue
        
        finally:
            # Закрываем соединение с БД
            db.close()
        
        return {
            "success": True,
            "processed_count": len(processed_locations),
            "total_records": len(data),
            "errors_count": len(errors),
            "errors": errors,
            "processed_locations": processed_locations,
            "processing_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "json_file": json_file_path
        }

def process_client_file_complete(file_path: str, user_id: int = 1) -> Dict[str, Any]:
    """
    Полный процесс обработки файла клиентов: загрузка -> JSON -> БД
    
    Args:
        file_path: Путь к исходному файлу
        user_id: ID пользователя
    
    Returns:
        Dict с полными результатами обработки
    """
    try:
        # Шаг 1: Загружаем файл
        upload_result = upload_client_file(file_path, user_id)
        if not upload_result["success"]:
            return upload_result
        
        # Шаг 2: Конвертируем в JSON
        json_result = convert_to_json(file_path, upload_result["json_file"])
        if not json_result["success"]:
            return json_result
        
        # Шаг 3: Обрабатываем JSON и сохраняем в БД
        db_result = process_json_to_locations(json_result["json_file"], user_id)
        if not db_result["success"]:
            return db_result
        
        # Очищаем временные файлы
        try:
            if os.path.exists(json_result["json_file"]):
                os.remove(json_result["json_file"])
        except:
            pass
        
        return {
            "success": True,
            "message": "Файл успешно обработан и загружен в базу данных",
            "upload_info": upload_result,
            "json_info": json_result,
            "database_info": db_result,
            "total_processed": db_result["processed_count"],
            "total_errors": db_result["errors_count"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

# Пример использования
if __name__ == "__main__":
    # Тестируем функции
    test_file = "Книга2.xlsx"  # Свежий тестовый CSV файл с ID 200+
    
    if os.path.exists(test_file):
        print("=== Тестирование обработки файла ===")
        
        # Полная обработка
        result = process_client_file_complete(test_file, user_id=3)
        
        if result["success"]:
            print(f"✅ Успешно обработано: {result['total_processed']} записей")
            print(f"❌ Ошибок: {result['total_errors']}")
            if result['total_errors'] > 0:
                print("Ошибки:")
                for error in result['database_info']['errors']:
                    print(f"  - {error}")
        else:
            print(f"❌ Ошибка: {result['error']}")
    else:
        print("Файл для тестирования не найден")
