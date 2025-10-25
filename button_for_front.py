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
import hashlib

# Импортируем настройки базы данных
from database_remote import get_db, DatabaseManager, session

def hash_password(password: str) -> str:
    """
    Хеширует пароль с использованием SHA-256
    
    Args:
        password: Пароль в открытом виде
    
    Returns:
        Хешированный пароль
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(first_name: str, last_name: str, phone_number: str, password: str) -> Dict[str, Any]:
    """
    Регистрирует нового пользователя в системе
    
    Args:
        first_name: Имя пользователя
        last_name: Фамилия пользователя
        phone_number: Номер телефона
        password: Пароль в открытом виде (будет захеширован)
    
    Returns:
        Dict с результатами регистрации
    """
    try:
        # Подключаемся к базе данных
        from database_remote import SessionLocal
        db = SessionLocal()
        db_manager = DatabaseManager(db)
        
        try:
            # Проверяем, не существует ли уже пользователь с таким номером телефона
            existing_users = db_manager.get_all_users()
            for user in existing_users:
                if user.phone_number == phone_number:
                    return {
                        "success": False,
                        "error": f"Пользователь с номером телефона {phone_number} уже существует",
                        "phone_number": phone_number
                    }
            
            # Хешируем пароль
            password_hash = hash_password(password)
            
            # Определяем следующий user_id
            max_user_id = max([user.user_id for user in existing_users]) if existing_users else 0
            next_user_id = max_user_id + 1
            
            # Создаем данные пользователя
            user_data = {
                "user_id": next_user_id,
                "first_name": first_name,
                "last_name": last_name,
                "phone_number": phone_number,
                "password_hash": password_hash
            }
            
            # Сохраняем пользователя в базу данных
            new_user = db_manager.create_user(user_data)
            
            return {
                "success": True,
                "message": f"Пользователь {first_name} {last_name} успешно зарегистрирован",
                "user_id": new_user.user_id,
                "first_name": new_user.first_name,
                "last_name": new_user.last_name,
                "phone_number": new_user.phone_number,
                "registration_time": datetime.now().isoformat()
            }
            
        finally:
            # Закрываем соединение с БД
            db.close()
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "first_name": first_name,
            "last_name": last_name,
            "phone_number": phone_number
        }

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

def mark_location_visited(user_id: int, location_id: int) -> Dict[str, Any]:
    """
    Отмечает, что пользователь посетил указанную локацию
    
    Args:
        user_id: ID пользователя
        location_id: ID локации для отметки
    
    Returns:
        Dict с результатами отметки
    """
    try:
        # Подключаемся к базе данных
        from database_remote import SessionLocal
        db = SessionLocal()
        db_manager = DatabaseManager(db)
        
        try:
            # Проверяем, что локация принадлежит данному пользователю
            user_locations = db_manager.get_locations_by_user(user_id)
            target_location = None
            
            for location in user_locations:
                if location.location_id == location_id:
                    target_location = location
                    break
            
            if not target_location:
                return {
                    "success": False,
                    "error": f"Локация с ID {location_id} не найдена для пользователя {user_id}",
                    "user_id": user_id,
                    "location_id": location_id
                }
            
            # Сохраняем исходный статус
            original_status = target_location.is_active
            
            # Определяем новый статус (переключаем: True -> False, False -> True)
            new_status = not original_status
            
            # Обновляем статус локации
            updated_location = db_manager.update_location_status(location_id, new_status)
            
            if updated_location:
                status_text = "отмечена как посещенная" if new_status else "отмечена как не посещенная"
                return {
                    "success": True,
                    "message": f"Локация {updated_location.address} {status_text}",
                    "user_id": user_id,
                    "location_id": location_id,
                    "address": updated_location.address,
                    "is_active": updated_location.is_active,
                    "previous_status": original_status,
                    "new_status": new_status,
                    "action_time": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Не удалось обновить статус локации",
                    "user_id": user_id,
                    "location_id": location_id
                }
            
        finally:
            # Закрываем соединение с БД
            db.close()
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "location_id": location_id
        }

def get_user_available_locations(user_id: int) -> Dict[str, Any]:
    """
    Получает список всех доступных локаций для пользователя
    
    Args:
        user_id: ID пользователя
    
    Returns:
        Dict со списком локаций пользователя
    """
    try:
        # Подключаемся к базе данных
        from database_remote import SessionLocal
        db = SessionLocal()
        db_manager = DatabaseManager(db)
        
        try:
            # Получаем все локации пользователя
            user_locations = db_manager.get_locations_by_user(user_id)
            
            locations_list = []
            for location in user_locations:
                locations_list.append({
                    "location_id": location.location_id,
                    "original_id": location.original_id,
                    "address": location.address,
                    "latitude": float(location.latitude),
                    "longitude": float(location.longitude),
                    "is_active": location.is_active,
                    "work_start": location.work_start,
                    "work_end": location.work_end,
                    "lunch_start": location.lunch_start,
                    "lunch_end": location.lunch_end,
                    "client_level": location.client_level
                })
            
            return {
                "success": True,
                "user_id": user_id,
                "total_locations": len(locations_list),
                "visited_locations": len([loc for loc in locations_list if loc["is_active"]]),
                "available_locations": len([loc for loc in locations_list if not loc["is_active"]]),
                "locations": locations_list,
                "query_time": datetime.now().isoformat()
            }
            
        finally:
            # Закрываем соединение с БД
            db.close()
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id
        }

def delete_user_locations(user_id: int) -> Dict[str, Any]:
    """
    Удаляет все записи локаций для указанного пользователя
    
    Args:
        user_id: ID пользователя, чьи записи нужно удалить
    
    Returns:
        Dict с результатами удаления
    """
    try:
        # Подключаемся к базе данных
        from database_remote import SessionLocal
        db = SessionLocal()
        db_manager = DatabaseManager(db)
        
        try:
            # Получаем все локации пользователя перед удалением
            user_locations = db_manager.get_locations_by_user(user_id)
            locations_count = len(user_locations)
            
            if locations_count == 0:
                return {
                    "success": True,
                    "message": f"У пользователя {user_id} нет записей для удаления",
                    "deleted_count": 0,
                    "user_id": user_id
                }
            
            # Удаляем все локации пользователя
            deleted_locations = []
            for location in user_locations:
                deleted_location = db_manager.delete_location(location.location_id)
                if deleted_location:
                    deleted_locations.append({
                        "location_id": deleted_location.location_id,
                        "original_id": deleted_location.original_id,
                        "address": deleted_location.address,
                        "client_level": deleted_location.client_level
                    })
            
            return {
                "success": True,
                "message": f"Успешно удалено {len(deleted_locations)} записей для пользователя {user_id}",
                "deleted_count": len(deleted_locations),
                "user_id": user_id,
                "deleted_locations": deleted_locations,
                "deletion_time": datetime.now().isoformat()
            }
            
        finally:
            # Закрываем соединение с БД
            db.close()
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id
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
    
    # Тестируем получение доступных локаций для пользователя
    print("\n=== Тестирование получения локаций пользователя ===")
    locations_result = get_user_available_locations(user_id=1)
    
    if locations_result.get("success"):
        print(f"✅ Найдено локаций: {locations_result['total_locations']}")
        print(f"📍 Посещено: {locations_result['visited_locations']}")
        print(f"🎯 Доступно для посещения: {locations_result['available_locations']}")
        
        if locations_result['locations']:
            print("\nСписок локаций:")
            for loc in locations_result['locations'][:3]:  # Показываем первые 3
                status = "✅ Посещена" if loc['is_active'] else "⏳ Доступна"
                print(f"  - ID: {loc['location_id']}, Адрес: {loc['address']}, Статус: {status}")
    else:
        print(f"❌ Ошибка получения локаций: {locations_result.get('error')}")
    
    # Тестируем отметку посещения локации
    if locations_result.get("success") and locations_result['available_locations'] > 0:
        print("\n=== Тестирование отметки посещения локации ===")
        
        # Находим первую доступную локацию
        available_location = None
        for loc in locations_result['locations']:
            if not loc['is_active']:
                available_location = loc
                break
        
        if available_location:
            visit_result = mark_location_visited(
                user_id=1,
                location_id=available_location['location_id']
            )
            
            if visit_result.get("success"):
                print(f"✅ {visit_result['message']}")
                print(f"📍 Адрес: {visit_result['address']}")
                print(f"🕐 Время действия: {visit_result['action_time']}")
                print(f"📊 Статус: {'Посещена' if visit_result['is_active'] else 'Не посещена'}")
            else:
                print(f"❌ Ошибка отметки посещения: {visit_result.get('error')}")
        else:
            print("ℹ️ Нет доступных локаций для отметки посещения")
    
    # Повторно проверяем статус локаций после отметки
    print("\n=== Проверка статуса после отметки ===")
    updated_locations = get_user_available_locations(user_id=1)
    if updated_locations.get("success"):
        print(f"✅ Обновленная статистика:")
        print(f"📍 Посещено: {updated_locations['visited_locations']}")
        print(f"🎯 Доступно для посещения: {updated_locations['available_locations']}")

    '''# Тестируем регистрацию пользователя
    print("\n=== Тестирование регистрации пользователя ===")
    registration_result = register_user(
        first_name="Bbf",
        last_name="Иванов",
        phone_number="+790fjjnf784567",
        password="fjjfjf"
    )

    if registration_result.get("success"):
        print(
            f"✅ Пользователь зарегистрирован: ID {registration_result['user_id']}, Имя {registration_result['first_name']} {registration_result['last_name']}")
    else:
        print(f"❌ Ошибка регистрации: {registration_result.get('error')}")'''

    '''
    # Тестируем функции
    test_file = "Книга1.xlsx"  # Тестовый CSV файл
    
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
    '''
    
    '''
    # Тестируем функцию удаления
    print("\n=== Тестирование удаления записей пользователя ===")
    delete_result = delete_user_locations(user_id=2)
    
    if delete_result["success"]:
        print(f"✅ {delete_result['message']}")
        print(f"🗑️ Удалено записей: {delete_result['deleted_count']}")
        if delete_result['deleted_count'] > 0:
            print("Удаленные записи:")
            for loc in delete_result['deleted_locations']:
                print(f"  - ID: {loc['location_id']}, Original: {loc['original_id']}, Адрес: {loc['address']}")
    else:
        print(f"❌ Ошибка удаления: {delete_result['error']}")
'''