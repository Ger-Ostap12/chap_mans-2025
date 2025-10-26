# 📊 Структура данных

## 🗂️ Файлы данных

### **DATA (2).txt** - Клиенты Ростова-на-Дону
```
Формат: JSON-like структура
Количество: 47 клиентов
Содержит: координаты, адреса, статусы VIP/Стандарт
```

**Пример записи:**
```
{
  "id": 1,
  "address": "Ростов-на-Дону, ул. Большая Садовая, 1",
  "lat": 47.2225,
  "lon": 39.7203,
  "client_level": "VIP",
  "work_start": "09:00",
  "work_end": "18:00",
  "lunch_start": "13:00",
  "lunch_end": "14:00"
}
```

## 🧠 Модели данных

### **Client** - Модель клиента
```python
@dataclass
class Client:
    id: int                    # Уникальный ID клиента
    address: str              # Адрес клиента
    lat: float                # Широта
    lon: float                # Долгота
    client_level: ClientLevel # VIP или Стандарт
    work_start: str           # Начало рабочего дня (HH:MM)
    work_end: str             # Конец рабочего дня (HH:MM)
    lunch_start: str          # Начало обеда (HH:MM)
    lunch_end: str            # Конец обеда (HH:MM)
```

### **ClientLevel** - Уровень клиента
```python
class ClientLevel(Enum):
    VIP = "VIP"           # VIP клиент (30 мин обслуживания)
    REGULAR = "Стандарт"  # Обычный клиент (20 мин обслуживания)
```

### **Location** - Местоположение
```python
@dataclass
class Location:
    latitude: float    # Широта
    longitude: float   # Долгота
    address: str       # Полный адрес
    city: str          # Город
    country: str       # Страна
    accuracy: float    # Точность в метрах
    source: str        # Источник: GPS/IP/Manual
```

## 📡 API Форматы

### **Запрос оптимизации маршрутов**
```json
{
  "clients_data": [
    {
      "id": 1,
      "address": "Ростов-на-Дону, ул. Большая Садовая, 1",
      "lat": 47.2225,
      "lon": 39.7203,
      "client_level": "VIP",
      "work_start": "09:00",
      "work_end": "18:00",
      "lunch_start": "13:00",
      "lunch_end": "14:00"
    }
  ],
  "num_days": 3
}
```

### **Ответ оптимизации маршрутов**
```json
{
  "success": true,
  "num_days": 3,
  "total_clients": 47,
  "user_location": {
    "latitude": 47.2225,
    "longitude": 39.7203,
    "address": "Ростов-на-Дону, Россия",
    "source": "GPS"
  },
  "routes": [
    {
      "day": 1,
      "clients": [
        {
          "id": 0,
          "address": "Пользователь: Ростов-на-Дону",
          "lat": 47.2225,
          "lon": 39.7203,
          "client_level": "Стандарт",
          "arrival_time": "09:00",
          "service_time": 0,
          "departure_time": "09:00"
        },
        {
          "id": 1,
          "address": "Ростов-на-Дону, ул. Большая Садовая, 1",
          "lat": 47.2225,
          "lon": 39.7203,
          "client_level": "VIP",
          "arrival_time": "09:15",
          "service_time": 30,
          "departure_time": "09:45"
        }
      ],
      "total_time": "6 часов 45 минут",
      "total_distance": "45.2 км",
      "tomtom_route": {
        "summary": {
          "lengthInMeters": 4520,
          "travelTimeInSeconds": 900,
          "trafficDelayInSeconds": 120
        }
      }
    }
  ]
}
```

## 🧠 ANN Модель - Входные данные

### **Признаки для обучения**
```python
features = [
    'latitude',           # Широта клиента
    'longitude',          # Долгота клиента
    'is_vip',            # VIP статус (0/1)
    'work_start_hour',   # Час начала работы (0-23)
    'work_end_hour',     # Час окончания работы (0-23)
    'lunch_start_hour',  # Час начала обеда (0-23)
    'lunch_end_hour',    # Час окончания обеда (0-23)
    'current_hour'       # Текущий час (0-23)
]
```

### **Целевые переменные**
```python
targets = [
    'trip_time',         # Время поездки в минутах
    'priority_score',    # Приоритет клиента (0-1)
    'route_score'        # Оценка маршрута (0-1)
]
```

## 📊 Данные обучения

### **NYC Taxi Data**
```
Файл: taxi_trip_data.csv
Записей: 10,000+
Столбцы: pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, trip_time
```

### **Синтетические данные**
```
Файл: combined_training_data.csv
Записей: 41,000+
Содержит: координаты + временные окна + VIP статусы
```

### **TomTom API данные**
```
Записей: 2,162 пары
Источник: Реальные запросы к TomTom API
Содержит: точные расстояния и время в пути
```

## 🔄 Поток данных

### **1. Загрузка клиентов**
```
DATA (2).txt → load_clients_from_file() → List[Client]
```

### **2. Определение местоположения**
```
GPS/IP/Адрес → LocationDetector → Location
```

### **3. Оптимизация маршрутов**
```
List[Client] → ANN Model → TomTom API → Route
```

### **4. Экспорт для фронтенда**
```
Route → export_routes_to_json() → frontend_routes.json
```

## 📈 Метрики и статистика

### **Производительность модели**
```python
metrics = {
    'mse': 0.08,           # Среднеквадратичная ошибка
    'mae': 0.15,           # Средняя абсолютная ошибка
    'r2_score': 0.92,      # Коэффициент детерминации
    'training_time': '2h', # Время обучения
    'epochs': 50           # Количество эпох
}
```

### **Статистика клиентов**
```python
client_stats = {
    'total_clients': 47,
    'vip_clients': 3,      # 6.4%
    'regular_clients': 44, # 93.6%
    'avg_service_time': 21.3, # минуты
    'coverage_area': 'Ростов-на-Дону'
}
```

## 🔧 Конфигурация

### **Временные параметры**
```python
time_config = {
    'work_start': '09:00',    # Начало рабочего дня
    'work_end': '18:00',      # Конец рабочего дня
    'lunch_start': '13:00',   # Начало обеда
    'lunch_end': '14:00',     # Конец обеда
    'vip_service_time': 30,   # Время обслуживания VIP (мин)
    'regular_service_time': 20 # Время обслуживания обычных (мин)
}
```

### **API параметры**
```python
api_config = {
    'tomtom_api_key': 'required',
    'max_retries': 3,
    'timeout': 10,
    'rate_limit': 2500,  # запросов/день
    'ssl_verify': False  # для решения SSL проблем
}
```

## 📝 Логирование

### **Формат логов**
```
[ВРЕМЯ] [УРОВЕНЬ] [МОДУЛЬ] СООБЩЕНИЕ

Примеры:
2024-01-15 10:30:15 INFO unified_route_system ✅ Маршрут построен: 3 дня
2024-01-15 10:30:16 INFO location_detector 📍 Местоположение: Ростов-на-Дону
2024-01-15 10:30:17 INFO api_endpoint 🚗 API запрос: /optimize_routes_from_location
```

### **Уровни логирования**
- `INFO` - Информационные сообщения
- `WARNING` - Предупреждения
- `ERROR` - Ошибки
- `DEBUG` - Отладочная информация
