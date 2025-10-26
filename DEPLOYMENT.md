# 🚀 Инструкции по развертыванию

## 📋 Предварительные требования

### 1. **Python 3.8+**
```bash
python --version
# Должно быть 3.8 или выше
```

### 2. **TomTom API Key**
- Зарегистрируйтесь на [developer.tomtom.com](https://developer.tomtom.com)
- Получите API ключ
- Лимит: 2500 запросов/день (бесплатно)

### 3. **Зависимости**
```bash
pip install -r requirements.txt
```

## 🔧 Настройка

### 1. **Переменные окружения**
```bash
# Windows
set TOMTOM_API_KEY=your_api_key_here

# Linux/Mac
export TOMTOM_API_KEY=your_api_key_here
```

### 2. **Проверка подключения**
```bash
python tomtom_diagnostics.py
```

### 3. **Тестирование системы**
```bash
python test_unified_system.py
```

## 🌐 Запуск API сервера

### 1. **Локальный запуск**
```bash
python api_endpoint.py
# Сервер будет доступен на http://localhost:8000
```

### 2. **С uvicorn**
```bash
uvicorn api_endpoint:app --reload --host 0.0.0.0 --port 8000
```

### 3. **Проверка работоспособности**
```bash
curl http://localhost:8000/health
```

## 📱 Интеграция с фронтендом

### 1. **Пример использования**
```javascript
// См. frontend_example.js
const optimizer = new RouteOptimizer('http://localhost:8000');

// Определение местоположения
await optimizer.detectUserLocation();

// Оптимизация маршрутов
const routes = await optimizer.optimizeRoutesFromLocation(clients, 3);
```

### 2. **API Endpoints**
- `POST /optimize_routes_from_location` - Оптимизация маршрутов
- `POST /update_location` - Обновление местоположения
- `POST /mark_visited` - Отметка клиента как посещенного
- `GET /location_suggestions` - Поиск адресов

## 🐳 Docker развертывание

### 1. **Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api_endpoint:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. **Запуск**
```bash
docker build -t route-optimizer .
docker run -p 8000:8000 -e TOMTOM_API_KEY=your_key route-optimizer
```

## ☁️ Облачное развертывание

### 1. **Heroku**
```bash
# Procfile
web: uvicorn api_endpoint:app --host 0.0.0.0 --port $PORT

# Переменные окружения
heroku config:set TOMTOM_API_KEY=your_key
```

### 2. **AWS/GCP/Azure**
- Используйте Docker контейнер
- Настройте переменные окружения
- Настройте load balancer
- Мониторинг и логирование

## 📊 Мониторинг

### 1. **Логи**
```python
# В коде уже есть логирование
print(f"✅ Маршрут построен: {len(routes)} дней")
print(f"📍 Местоположение обновлено: {location.address}")
```

### 2. **Метрики**
- Время ответа API
- Количество запросов к TomTom
- Использование памяти
- Ошибки API

### 3. **Health Check**
```bash
curl http://localhost:8000/health
# Ответ: {"status": "healthy", "tomtom_api_key": "configured"}
```

## 🔒 Безопасность

### 1. **API ключи**
- Никогда не коммитьте API ключи в код
- Используйте переменные окружения
- Ротируйте ключи регулярно

### 2. **Валидация данных**
```python
# В API уже есть валидация
if not latitude or not longitude:
    raise HTTPException(status_code=400, detail="Требуются latitude и longitude")
```

### 3. **Rate Limiting**
- TomTom API: 2500 запросов/день
- Добавьте кэширование для часто используемых маршрутов

## 🐛 Отладка

### 1. **Проблемы с SSL**
```python
# Уже решено в коде
session.verify = False
```

### 2. **Ошибки TomTom API**
```python
# Проверьте API ключ
python tomtom_diagnostics.py

# Проверьте лимиты
# 2500 запросов/день
```

### 3. **Проблемы с моделью**
```python
# Проверьте наличие файла
ls -la best_unified_model.pth

# Переобучите если нужно
python unified_route_system.py
```

## 📈 Масштабирование

### 1. **Горизонтальное масштабирование**
- Запустите несколько экземпляров API
- Используйте load balancer
- Кэшируйте результаты TomTom API

### 2. **Оптимизация производительности**
- Кэширование маршрутов
- Асинхронные запросы к TomTom
- Пулинг соединений

### 3. **Мониторинг ресурсов**
- CPU: 1-2 ядра
- RAM: 500MB-1GB
- Диск: 100MB (модель + данные)

## 🔄 Обновления

### 1. **Обновление модели**
```bash
# Переобучите модель
python unified_route_system.py

# Перезапустите сервер
uvicorn api_endpoint:app --reload
```

### 2. **Обновление API**
```bash
git pull origin main
pip install -r requirements.txt
uvicorn api_endpoint:app --reload
```

## 📞 Поддержка

### 1. **Логи ошибок**
```bash
# Проверьте логи сервера
tail -f server.log

# Проверьте статус API
curl http://localhost:8000/health
```

### 2. **Тестирование**
```bash
# Полное тестирование
python test_unified_system.py

# Тестирование местоположения
python location_detector.py
```

### 3. **Контакты**
- Документация: README.md
- Архитектура: ARCHITECTURE.md
- Примеры: frontend_example.js
