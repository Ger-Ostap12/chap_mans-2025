# 🏗️ Архитектура системы маршрутизации

## 📋 Обзор системы

Система оптимизации маршрутов, объединяющая ANN модель, TomTom API и динамическое планирование маршрутов.

## 🧠 Основные компоненты

### 1. **UnifiedRouteSystem** (`unified_route_system.py`)
**Главный класс системы**

```python
class UnifiedRouteSystem:
    def __init__(self, tomtom_api_key, model_path=None):
        self.model = AttentionRouteOptimizer()  # ANN модель
        self.location_detector = LocationDetector()  # Определение местоположения
        self.user_location = None  # Текущее местоположение пользователя
        self.visited_clients = set()  # Посещенные клиенты
        self.current_time = 9.0  # Текущее время (часы)
```

**Ключевые методы:**
- `get_unified_route()` - построение оптимизированного маршрута
- `mark_client_visited()` - отметка клиента как посещенного
- `update_user_location()` - обновление местоположения пользователя
- `recalculate_routes_from_new_location()` - пересчет маршрутов

### 2. **AttentionRouteOptimizer** (ANN модель)
**Нейронная сеть для оптимизации маршрутов**

```python
class AttentionRouteOptimizer(nn.Module):
    def __init__(self):
        self.input_layer = nn.Linear(8, 64)  # Входные признаки
        self.attention = nn.MultiheadAttention(64, 8)  # Attention механизм
        self.output_layer = nn.Linear(64, 3)  # Выходы: score, time, priority
```

**Обучена на:**
- NYC taxi данных (41,000+ пар)
- Синтетических данных с временными окнами
- Реальных данных TomTom API (2,162 пары)

### 3. **LocationDetector** (`location_detector.py`)
**Определение местоположения пользователя**

```python
class LocationDetector:
    def get_best_location(self, gps_coords=None, ip_address=None, manual_address=None):
        # Приоритет: GPS > Manual > IP > Default
```

**Поддерживает:**
- GPS координаты (точность 10м)
- IP геолокацию (точность 5км)
- Адресное геокодирование (точность 100м)
- Поиск предложений адресов

### 4. **FastAPI Server** (`api_endpoint.py`)
**REST API для фронтенда**

```python
app = FastAPI()

@app.post("/optimize_routes_from_location")
async def optimize_routes_from_location(clients_data, num_days):
    # Оптимизация маршрутов от местоположения пользователя

@app.post("/update_location")
async def update_location(location_data):
    # Обновление местоположения (с карты)

@app.post("/mark_visited")
async def mark_visited(client_id, actual_service_time):
    # Отметка клиента как посещенного
```

## 🔄 Поток данных

### 1. **Инициализация**
```
Frontend → /set_location → LocationDetector → UnifiedRouteSystem
```

### 2. **Построение маршрута**
```
Frontend → /optimize_routes_from_location → UnifiedRouteSystem → ANN Model + TomTom API
```

### 3. **Динамическое обновление**
```
Frontend → /update_location → UnifiedRouteSystem → recalculate_routes_from_new_location
```

### 4. **Отметка посещения**
```
Frontend → /mark_visited → UnifiedRouteSystem → recalculate_remaining_routes
```

## 📊 Модель данных

### **Client**
```python
@dataclass
class Client:
    id: int
    address: str
    lat: float
    lon: float
    client_level: ClientLevel  # VIP/Стандарт
    work_start: str  # "09:00"
    work_end: str    # "18:00"
    lunch_start: str # "13:00"
    lunch_end: str   # "14:00"
```

### **Location**
```python
@dataclass
class Location:
    latitude: float
    longitude: float
    address: str
    city: str
    country: str
    accuracy: float  # Точность в метрах
    source: str      # GPS/IP/Manual
```

## 🎯 Алгоритм оптимизации

### 1. **Распределение по дням**
```python
def distribute_clients_by_days(clients, num_days):
    # Равномерное распределение с учетом VIP приоритетов
    # Пример: 20 клиентов на 3 дня = 7+7+6
```

### 2. **Оптимизация маршрута дня**
```python
def optimize_route_for_day(clients):
    # 1. ANN модель предсказывает приоритеты
    # 2. TomTom API дает реальные расстояния
    # 3. Учитываются временные окна клиентов
    # 4. Строится оптимальный порядок посещения
```

### 3. **Динамический пересчет**
```python
def recalculate_remaining_routes():
    # 1. Удаляем посещенных клиентов
    # 2. Обновляем текущее время
    # 3. Пересчитываем оставшиеся маршруты
    # 4. TomTom API дает актуальное время
```

## 🔧 Интеграция с TomTom API

### **Получение маршрута**
```python
def get_tomtom_route(self, start_lat, start_lon, end_lat, end_lon):
    url = "https://api.tomtom.com/routing/1/calculateRoute"
    params = {
        'key': self.tomtom_api_key,
        'waypoints': f"{start_lat},{start_lon}:{end_lat},{end_lon}",
        'traffic': 'true',  # Учет трафика
        'routeType': 'fastest'
    }
```

### **Обработка ошибок**
```python
# SSL проблемы решены через session.verify = False
# Retry логика для API лимитов
# Fallback на синтетические данные
```

## 📈 Производительность

### **Обучение модели**
- **Данные**: 41,000+ пар координат
- **Эпохи**: 50 (с early stopping)
- **Время**: ~2 часа на CPU
- **Точность**: MSE < 0.1

### **Работа в продакшене**
- **47 клиентов**: < 5 секунд
- **TomTom API**: 2-3 секунды на запрос
- **Память**: ~500MB
- **CPU**: 1-2 ядра

## 🚀 Развертывание

### **Требования**
```bash
pip install -r requirements.txt
export TOMTOM_API_KEY="your_key"
```

### **Запуск**
```bash
# API сервер
uvicorn api_endpoint:app --reload

# Тестирование
python test_unified_system.py
```

### **Мониторинг**
- Логи API запросов
- Метрики производительности
- Статус TomTom API
- Использование памяти

## 🔒 Безопасность

- API ключи в переменных окружения
- Валидация входных данных
- Обработка ошибок API
- Rate limiting для TomTom API

## 📝 Логирование

```python
print(f"📍 Определение местоположения: {location.address}")
print(f"✅ Клиент {client_id} отмечен как посещенный")
print(f"🔄 Пересчет маршрутов с нового местоположения")
```
