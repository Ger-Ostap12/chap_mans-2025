"""
Обучение ML модели на данных из NYC
Использует Attention-based Neural Network для оптимизации маршрутов
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import sys

class RouteOptimizationDataset(Dataset):
    """
    Датасет для обучения модели оптимизации маршрутов на NYC данных
    """

    def __init__(self, data_dir: str = "DS"):
        self.data_dir = data_dir
        self.trips = self._load_nyc_data()

    def _load_nyc_data(self):
        """
        Загрузка объединенных данных (NYC + синтетические времена)
        """
        # Сначала проверяем объединенные данные
        combined_data_path = "combined_training_data.csv"

        if os.path.exists(combined_data_path):
            print("📊 Загрузка объединенных данных (NYC + времена)...")
            return self._load_combined_data(combined_data_path)
        else:
            print("📊 Загрузка только NYC данных (без временных параметров)...")
            return self._load_nyc_only_data()

    def _load_combined_data(self, data_path: str):
        """
        Загрузка объединенных данных с временными параметрами
        """
        try:
            df = pd.read_csv(data_path)
            print(f"📋 Колонки в данных: {list(df.columns)}")

            trips = []
            for _, row in df.iterrows():
                trip = {
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'is_vip': bool(row['is_vip']),
                    'work_start_hour': float(row['work_start_hour']),
                    'work_end_hour': float(row['work_end_hour']),
                    'lunch_start_hour': float(row['lunch_start_hour']),
                    'lunch_end_hour': float(row['lunch_end_hour']),
                    'timestamp': np.random.randint(0, 86400),
                    'distance': 1.0,
                    'time': 30.0,
                    'fare': 10.0
                }
                trips.append(trip)

            print(f"✅ Загружено {len(trips)} записей с временными параметрами")
            return trips

        except Exception as e:
            print(f"❌ Ошибка загрузки объединенных данных: {e}")
            return self._load_nyc_only_data()

    def _load_nyc_only_data(self):
        """
        Загрузка только NYC данных (старый метод)
        """
        trip_data_path = os.path.join(self.data_dir, "taxi_trip_data.csv")

        if not os.path.exists(trip_data_path):
            print(f"❌ Файл данных не найден: {trip_data_path}")
            return []

        print("📊 Загрузка NYC данных...")

        # Читаем данные порциями для экономии памяти
        trips = []
        chunk_size = 10000  # Уменьшаем размер чанка

        try:
            for chunk in pd.read_csv(trip_data_path, chunksize=chunk_size):
                # Анализируем структуру данных
                if len(trips) == 0:
                    print(f"📋 Колонки в данных: {list(chunk.columns)}")

                # Обрабатываем только первые несколько чанков для обучения
                if len(trips) >= 50000:  # Ограничиваем до 50k поездок
                    break

                # Используем данные из NYC структуры
                for _, row in chunk.iterrows():
                    try:
                        # Извлекаем данные из NYC структуры
                        pickup_location = row.get('pickup_location_id', None)
                        dropoff_location = row.get('dropoff_location_id', None)
                        trip_distance = row.get('trip_distance', 0)
                        fare_amount = row.get('fare_amount', 0)

                        # Проверяем валидность данных
                        if (pd.notna(pickup_location) and pd.notna(dropoff_location) and
                            pickup_location > 0 and dropoff_location > 0):

                            # Создаем поездку с координатами из зон
                            # Используем ID зон как координаты (упрощение)
                            trip = {
                                'latitude': pickup_location / 1000.0,  # Нормализуем ID зоны
                                'longitude': dropoff_location / 1000.0,  # Нормализуем ID зоны
                                'is_vip': np.random.choice([True, False]),  # Случайный VIP
                                'work_start_hour': 0.33,  # 08:00 нормализованное
                                'work_end_hour': 0.75,   # 18:00 нормализованное
                                'lunch_start_hour': 0.54, # 13:00 нормализованное
                                'lunch_end_hour': 0.58,   # 14:00 нормализованное
                                'timestamp': np.random.randint(0, 86400),  # случайное время
                                'distance': float(trip_distance) if pd.notna(trip_distance) else 1.0,
                                'time': np.random.uniform(5, 120),  # случайное время поездки
                                'fare': float(fare_amount) if pd.notna(fare_amount) else 10.0
                            }
                            trips.append(trip)

                    except Exception as e:
                        continue

                print(f"📊 Обработано {len(trips)} поездок...")

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return []

        print(f"✅ Загружено {len(trips)} поездок из NYC")
        return trips

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        """
        Получение одного элемента датасета
        """
        # Создаем группу из 8 случайных поездок для задачи оптимизации
        group_size = 8
        start_idx = (idx * group_size) % len(self.trips)

        # Берем группу поездок
        group_trips = []
        for i in range(group_size):
            trip_idx = (start_idx + i) % len(self.trips)
            group_trips.append(self.trips[trip_idx])

        # Подготавливаем данные для модели
        features = self._prepare_features(group_trips)
        targets = self._prepare_targets(group_trips)

        return {
            'features': torch.FloatTensor(features),
            'targets': torch.LongTensor(targets),
            'trip_group_id': f"group_{idx}"
        }

    def _prepare_features(self, trips):
        """
        Подготовка признаков для модели (7 параметров)
        """
        features = []

        for trip in trips:
            # Основные признаки поездки (7 параметров)
            trip_features = [
                trip['latitude'],
                trip['longitude'],
                float(trip['is_vip']),
                trip['work_start_hour'],
                trip['work_end_hour'],
                trip['lunch_start_hour'],
                trip['lunch_end_hour']
            ]

            features.append(trip_features)

        return np.array(features)

    def _prepare_targets(self, trips):
        """
        Подготовка целевых значений (оптимальный порядок посещения)
        """
        # Создаем простой оптимальный маршрут (ближайший сосед)
        targets = [0] * len(trips)
        visited = [False] * len(trips)
        current = 0
        visited[0] = True

        for step in range(1, len(trips)):
            min_distance = float('inf')
            nearest = 0

            for i in range(len(trips)):
                if not visited[i]:
                    # Вычисляем расстояние
                    dist = np.sqrt(
                        (trips[current]['latitude'] - trips[i]['latitude'])**2 +
                        (trips[current]['longitude'] - trips[i]['longitude'])**2
                    )

                    if dist < min_distance:
                        min_distance = dist
                        nearest = i

            targets[nearest] = step
            visited[nearest] = True
            current = nearest

        return targets

class AttentionRouteOptimizer(nn.Module):
    """
    Attention-based Neural Network для оптимизации маршрутов
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Входной слой
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Attention слои
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # Нормализация
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Выходные слои
        self.route_output = nn.Linear(hidden_dim, 1)
        self.time_output = nn.Linear(hidden_dim, 1)
        self.priority_output = nn.Linear(hidden_dim, 1)

        # Активация
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        """
        Прямой проход модели
        """
        batch_size, seq_len, _ = x.shape

        # Входной слой
        x = self.input_layer(x)

        # Attention слои
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            # Self-attention
            attn_output, _ = attention(x, x, x, key_padding_mask=mask)
            x = norm(x + attn_output)
            x = self.activation(x)

        # Выходные предсказания
        route_scores = self.route_output(x)
        time_predictions = self.time_output(x)
        priority_scores = self.priority_output(x)

        return {
            'route_scores': route_scores.squeeze(-1),
            'time_predictions': time_predictions.squeeze(-1),
            'priority_scores': priority_scores.squeeze(-1)
        }

class ModelTrainer:
    """
    Тренер для обучения модели
    """

    def __init__(self, data_dir: str = "DS"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создаем модель
        self.model = AttentionRouteOptimizer().to(self.device)

        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Используем MSE для регрессии

        print(f"🖥️  Устройство: {self.device}")
        print(f"🧠 Модель: {sum(p.numel() for p in self.model.parameters())} параметров")

    def train_epoch(self, dataloader):
        """
        Обучение одной эпохи
        """
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)

            # Создаем маску для padding
            batch_size, seq_len = features.shape[:2]
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

            # Прямой проход
            outputs = self.model(features, mask)

            # Вычисляем потери (используем route_scores для регрессии)
            # Преобразуем targets в float для MSE
            targets_float = targets.float()
            loss = self.criterion(outputs['route_scores'], targets_float)

            # Обратное распространение
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """
        Валидация модели
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)

                # Создаем маску
                batch_size, seq_len = features.shape[:2]
                mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

                # Прямой проход
                outputs = self.model(features, mask)

                # Вычисляем потери (используем route_scores для регрессии)
                targets_float = targets.float()
                loss = self.criterion(outputs['route_scores'], targets_float)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, num_epochs: int = 50, start_epoch: int = 0):
        """
        Полное обучение модели
        """
        print("🚀 Начало обучения модели...")

        # Создаем датасет и даталоадер
        dataset = RouteOptimizationDataset(self.data_dir)

        if len(dataset) == 0:
            print("❌ Нет данных для обучения")
            return False

        # Разделяем на train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Пытаемся загрузить существующую модель
        best_val_loss = float('inf')
        if os.path.exists("best_model.pth"):
            print("📁 Загружаем существующую модель...")
            try:
                checkpoint = torch.load("best_model.pth", map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print("✅ Модель загружена успешно")
            except Exception as e:
                print(f"⚠️  Ошибка загрузки модели: {e}")
                print("🔄 Начинаем обучение с нуля")

        # Обучение
        patience = 10
        patience_counter = 0

        for epoch in range(start_epoch, num_epochs):
            # Обучение
            train_loss = self.train_epoch(train_loader)

            # Валидация
            val_loss = self.validate(val_loader)

            print(f"Эпоха {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем лучшую модель
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping на эпохе {epoch+1}")
                    break

        print("✅ Обучение завершено!")
        print(f"📁 Лучшая модель сохранена: best_model.pth")

        return True

    def save_model(self, path: str = "trained_model.pth"):
        """
        Сохранение обученной модели
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_dim': 7,
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 3
            }
        }, path)

        print(f"💾 Модель сохранена: {path}")

def main():
    """
    Основная функция обучения
    """
    print("🎯 Обучение ML модели на данных из NYC")
    print("=" * 60)

    # Проверяем наличие данных
    data_dir = "DS"
    trip_file = os.path.join(data_dir, "taxi_trip_data.csv")

    if not os.path.exists(trip_file):
        print(f"❌ Файл данных не найден: {trip_file}")
        print("💡 Убедитесь, что файлы NYC данных находятся в папке DS/")
        return

    # Создаем тренер
    trainer = ModelTrainer(data_dir)

    # Обучаем модель (продолжаем с 25-й эпохи до 30-й)
    success = trainer.train(num_epochs=30, start_epoch=25)

    if success:
        # Сохраняем финальную модель
        trainer.save_model("trained_model.pth")

        print("\n🎉 Модель обучена успешно!")
        print("📁 Файлы модели:")
        print("- best_model.pth (лучшая модель)")
        print("- trained_model.pth (финальная модель)")
        print("\n📋 Следующие шаги:")
        print("1. Используйте обученную модель для построения маршрутов")
        print("2. Модель готова к интеграции в ваше приложение")
        print("\n💡 Модель обучена на реальных данных из NYC!")
    else:
        print("\n❌ Ошибка при обучении модели")

if __name__ == "__main__":
    main()
