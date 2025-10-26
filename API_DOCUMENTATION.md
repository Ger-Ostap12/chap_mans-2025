# 📡 API Документация

## 🌐 Базовый URL
```
http://localhost:8000
```

## 🔑 Аутентификация
API ключ TomTom должен быть установлен в переменной окружения:
```bash
export TOMTOM_API_KEY="your_api_key_here"
```

## 📋 Endpoints

### 1. **Оптимизация маршрутов от местоположения пользователя**

**POST** `/optimize_routes_from_location`

**Описание:** Строит оптимизированный маршрут от текущего местоположения пользователя к клиентам.

**Тело запроса:**
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

**Ответ:**
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
        }
      ],
      "total_time": "6 часов 45 минут",
      "total_distance": "45.2 км"
    }
  ]
}
```

### 2. **Установка местоположения пользователя**

**POST** `/set_location`

**Описание:** Устанавливает местоположение пользователя (GPS, IP или адрес).

**Тело запроса:**
```json
{
  "gps_coords": [47.2225, 39.7203],
  "manual_address": "Ростов-на-Дону, Театральная площадь",
  "ip_address": "8.8.8.8"
}
```

**Ответ:**
```json
{
  "success": true,
  "location": {
    "latitude": 47.2225,
    "longitude": 39.7203,
    "address": "Ростов-на-Дону, Театральная площадь, Россия",
    "city": "Ростов-на-Дону",
    "country": "Россия",
    "accuracy": 10.0,
    "source": "GPS"
  },
  "message": "Местоположение определено: Ростов-на-Дону, Театральная площадь"
}
```

### 3. **Обновление местоположения (с карты)**

**POST** `/update_location`

**Описание:** Обновляет местоположение пользователя (например, при перетаскивании маркера на карте).

**Тело запроса:**
```json
{
  "latitude": 47.2500,
  "longitude": 39.7500
}
```

**Ответ:**
```json
{
  "success": true,
  "old_location": {
    "latitude": 47.2225,
    "longitude": 39.7203,
    "address": "Ростов-на-Дону, Театральная площадь"
  },
  "new_location": {
    "latitude": 47.2500,
    "longitude": 39.7500,
    "address": "Ростов-на-Дону, ул. Пушкинская, Россия",
    "city": "Ростов-на-Дону",
    "country": "Россия",
    "accuracy": 10.0,
    "source": "GPS"
  },
  "message": "Местоположение обновлено: Ростов-на-Дону, ул. Пушкинская"
}
```

### 4. **Пересчет маршрутов с нового местоположения**

**POST** `/recalculate_routes`

**Описание:** Пересчитывает маршруты с обновленного местоположения пользователя.

**Тело запроса:**
```json
{
  "clients_data": [
    {
      "id": 1,
      "address": "Ростов-на-Дону, ул. Большая Садовая, 1",
      "lat": 47.2225,
      "lon": 39.7203,
      "client_level": "VIP"
    }
  ],
  "num_days": 3
}
```

**Ответ:**
```json
{
  "success": true,
  "routes": {
    "success": true,
    "num_days": 3,
    "total_clients": 47,
    "routes": [...]
  },
  "message": "Маршруты пересчитаны с нового местоположения"
}
```

### 5. **Отметка клиента как посещенного**

**POST** `/mark_visited`

**Описание:** Отмечает клиента как посещенного и пересчитывает оставшиеся маршруты.

**Тело запроса:**
```json
{
  "client_id": 1,
  "actual_service_time": 25.0
}
```

**Ответ:**
```json
{
  "success": true,
  "visited_client": {
    "id": 1,
    "address": "Ростов-на-Дону, ул. Большая Садовая, 1",
    "planned_service_time": 30.0,
    "actual_service_time": 25.0,
    "time_saved": 5.0
  },
  "updated_time": "09:25",
  "remaining_clients": 46,
  "message": "Клиент 1 отмечен как посещенный"
}
```

### 6. **Поиск предложений адресов**

**GET** `/location_suggestions?q=Ростов-на-Дону&limit=5`

**Описание:** Получает предложения адресов для автодополнения.

**Параметры:**
- `q` (string) - поисковый запрос
- `limit` (int) - максимальное количество предложений (по умолчанию 5)

**Ответ:**
```json
{
  "success": true,
  "suggestions": [
    {
      "display_name": "Ростов-на-Дону, Ростовская область, Россия",
      "latitude": 47.2225,
      "longitude": 39.7203,
      "address": {
        "city": "Ростов-на-Дону",
        "state": "Ростовская область",
        "country": "Россия"
      },
      "importance": 0.8
    }
  ],
  "query": "Ростов-на-Дону"
}
```

### 7. **Получение текущего местоположения**

**GET** `/current_location`

**Описание:** Возвращает текущее местоположение пользователя.

**Ответ:**
```json
{
  "success": true,
  "location": {
    "latitude": 47.2225,
    "longitude": 39.7203,
    "address": "Ростов-на-Дону, Театральная площадь, Россия",
    "city": "Ростов-на-Дону",
    "country": "Россия",
    "accuracy": 10.0,
    "source": "GPS"
  }
}
```

### 8. **Проверка состояния системы**

**GET** `/health`

**Описание:** Проверяет состояние системы и конфигурацию.

**Ответ:**
```json
{
  "status": "healthy",
  "tomtom_api_key": "configured",
  "model_loaded": "best_unified_model.pth"
}
```

## 🔧 Коды ошибок

### **400 Bad Request**
```json
{
  "detail": "Требуются latitude и longitude"
}
```

### **500 Internal Server Error**
```json
{
  "detail": "Ошибка TomTom API: Max retries exceeded"
}
```

## 📊 Примеры использования

### **1. Полный цикл работы**

```javascript
// 1. Определение местоположения
const locationResponse = await fetch('/set_location', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ gps_coords: [47.2225, 39.7203] })
});

// 2. Оптимизация маршрутов
const routesResponse = await fetch('/optimize_routes_from_location', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    clients_data: clients,
    num_days: 3
  })
});

// 3. Обновление местоположения
const updateResponse = await fetch('/update_location', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    latitude: 47.2500,
    longitude: 39.7500
  })
});

// 4. Отметка клиента как посещенного
const visitedResponse = await fetch('/mark_visited', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    client_id: 1,
    actual_service_time: 25.0
  })
});
```

### **2. Обработка ошибок**

```javascript
try {
  const response = await fetch('/optimize_routes_from_location', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clients_data: clients, num_days: 3 })
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const result = await response.json();

  if (result.success) {
    console.log('Маршруты построены:', result.routes);
  } else {
    console.error('Ошибка:', result.message);
  }
} catch (error) {
  console.error('Ошибка API:', error.message);
}
```

## 🚀 Производительность

- **Время ответа**: 2-5 секунд для 47 клиентов
- **TomTom API**: 2-3 секунды на запрос
- **Память**: ~500MB
- **CPU**: 1-2 ядра

## 🔒 Безопасность

- Валидация входных данных
- Обработка ошибок API
- Rate limiting для TomTom API
- Безопасное хранение API ключей
