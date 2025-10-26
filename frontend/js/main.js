// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const fileInput = document.getElementById("fileInput");
const filePlaceholder = document.getElementById("filePlaceholder");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const removeFileBtn = document.getElementById("removeFileBtn");
const getGeoBtn = document.getElementById("getGeo");
const manualAddressInput = document.getElementById("manualAddress");
const geoError = document.getElementById("geoError");
const form = document.querySelector("form");
const getExportFileBtn = document.getElementById("getExportFile");

// Global variables
let currentLocation = null;
let uploadedFile = null;
let currentRoute = null;
let map = null;
let currentUser = null;
let currentUserId = 1; // Default user ID

// File upload handling with authentication check and automatic processing
fileInput.addEventListener("change", async function () {
  // Check if user is authenticated
  if (!currentUser) {
    // Clear file input
    this.value = '';
    
    // Show registration modal
    showNotification('Для загрузки файлов необходимо зарегистрироваться', 'warning');
    const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
    registerModal.show();
    return;
  }
  
  if (this.files && this.files.length > 0) {
    uploadedFile = this.files[0];
    fileName.textContent = `Файл: ${uploadedFile.name}`;
    fileName.classList.add("col-6");
    removeFileBtn.classList.add("col-6");
    filePlaceholder.classList.add("d-none");
    fileInfo.classList.remove("d-none");
    fileInfo.classList.add("d-flex");
    
    // Автоматически загружаем файл в базу данных
    try {
      console.log("Автоматическая загрузка файла в базу данных...");
      const uploadResult = await uploadAdvancedFile(uploadedFile);
      console.log("Файл автоматически загружен:", uploadResult);
      
      if (uploadResult.total_processed > 0) {
        showNotification(`Файл автоматически загружен в базу данных! Обработано записей: ${uploadResult.total_processed}`, 'success');
        
        // Обновляем данные на странице - всегда показываем данные из загруженного файла
        const locations = await getUserLocations(currentUserId);
        console.log("Полученные локации:", locations);
        if (locations.locations && locations.locations.length > 0) {
          updateTable(locations.locations);
          console.log(`Загружено ${locations.total_locations} локаций из файла`);
        } else {
          console.log("Нет локаций для отображения после загрузки файла");
          // Если файл загружен но нет данных, показываем таблицу по умолчанию
          showDefaultTable();
        }
      } else {
        showNotification("Файл загружен, но не было обработано записей", 'warning');
      }
    } catch (error) {
      console.error("Ошибка автоматической загрузки:", error);
      showNotification("Ошибка при автоматической загрузке файла: " + error.message, 'error');
    }
  }
});

removeFileBtn.addEventListener("click", async function () {
  // Check if user is authenticated
  if (!currentUser) {
    showNotification('Для удаления файлов необходимо зарегистрироваться', 'warning');
    const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
    registerModal.show();
    return;
  }
  
  // Подтверждение удаления
  if (!confirm("Вы уверены, что хотите удалить файл? Все данные из базы данных будут удалены!")) {
    return;
  }
  
  try {
    // Удаляем все данные пользователя из базы данных
    console.log("Удаление всех данных пользователя из базы данных...");
    const deleteResult = await deleteUserLocations(currentUserId);
    console.log("Данные удалены:", deleteResult);
    
    if (deleteResult.success) {
      showNotification(`Все данные удалены из базы данных! Удалено записей: ${deleteResult.deleted_count}`, 'success');
      
      // Очищаем таблицу
      const tbody = document.querySelector('table tbody');
      tbody.innerHTML = '';
      
      // Очищаем карту
      if (map) {
        map.eachLayer(function(layer) {
          if (layer instanceof L.Marker || layer instanceof L.Polyline) {
            map.removeLayer(layer);
          }
        });
      }
    } else {
      showNotification("Ошибка при удалении данных: " + deleteResult.error, 'error');
    }
  } catch (error) {
    console.error("Ошибка удаления данных:", error);
    showNotification("Ошибка при удалении данных: " + error.message, 'error');
  }
  
  // сбрасываем input
  fileInput.value = "";
  uploadedFile = null;

  // возвращаем интерфейс в исходное состояние
  filePlaceholder.classList.remove("d-none");
  fileInfo.classList.add("d-none");
});

// Geolocation handling
getGeoBtn.addEventListener("click", function () {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      function (position) {
        currentLocation = {
          lat: position.coords.latitude,
          lon: position.coords.longitude
        };
        manualAddressInput.value = `${currentLocation.lat.toFixed(6)}, ${currentLocation.lon.toFixed(6)}`;
        manualAddressInput.disabled = false;
        geoError.classList.add("d-none");
        console.log("Геолокация получена:", currentLocation);
      },
      function (error) {
        console.error("Ошибка геолокации:", error);
        geoError.classList.remove("d-none");
        manualAddressInput.disabled = false;
      }
    );
  } else {
    console.error("Геолокация не поддерживается");
    geoError.classList.remove("d-none");
    manualAddressInput.disabled = false;
  }
});

// Form submission with authentication check
form.addEventListener("submit", async function (e) {
  e.preventDefault();
  
  // Check if user is authenticated
  if (!currentUser) {
    showNotification('Для загрузки файлов необходимо зарегистрироваться', 'warning');
    const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
    registerModal.show();
    return;
  }
  
  if (!uploadedFile) {
    alert("Пожалуйста, выберите файл с данными клиентов");
    return;
  }

  if (!currentLocation && !manualAddressInput.value.trim()) {
    alert("Пожалуйста, разрешите геолокацию или введите адрес вручную");
    return;
  }

  try {
    // Upload file using advanced processing
    const uploadResult = await uploadAdvancedFile(uploadedFile);
    console.log("Файл загружен и обработан:", uploadResult);

    // Show processing results
    if (uploadResult.total_processed > 0) {
      alert(`Файл успешно обработан!\nОбработано записей: ${uploadResult.total_processed}\nОшибок: ${uploadResult.total_errors}`);
    }

    // Get locations from uploaded file (using button_for_front.py data)
    const locations = await getUserLocations(currentUserId);
    console.log("Локации получены:", locations);

    // Build route
    const startLocation = currentLocation || await geocodeAddress(manualAddressInput.value);
    const clientIds = locations.locations.map(loc => loc.location_id);
    
    // Create route using location data
    const routeWaypoints = locations.locations.map((location, index) => ({
      order: index + 1,
      type: "client",
      lat: location.latitude,
      lon: location.longitude,
      address: location.address,
      client_id: location.location_id,
      client_level: location.client_level
    }));

    // Add start point
    routeWaypoints.unshift({
      order: 0,
      type: "start",
      lat: startLocation.lat,
      lon: startLocation.lon,
      address: "Начальная точка",
      client_id: null
    });

    // Create route object
    const route = {
      route_id: "manual_" + Date.now(),
      total_distance: calculateTotalDistance(routeWaypoints),
      total_time: calculateTotalTime(calculateTotalDistance(routeWaypoints)),
      waypoints: routeWaypoints,
      optimized: true
    };

    console.log("Маршрут построен:", route);
    
    currentRoute = route;
    displayRoute(route);
    updateTable(locations.locations);

  } catch (error) {
    console.error("Ошибка при построении маршрута:", error);
    alert("Произошла ошибка при построении маршрута. Проверьте консоль для подробностей.");
  }
});

// Export functionality
getExportFileBtn.addEventListener("click", async function () {
  if (!currentRoute) {
    alert("Сначала постройте маршрут");
    return;
  }

  try {
    const exportData = await exportReport("csv");
    console.log("Отчет экспортирован:", exportData);
    
    // Create download link
    const blob = new Blob([exportData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `route_report_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
  } catch (error) {
    console.error("Ошибка при экспорте:", error);
    alert("Произошла ошибка при экспорте отчета");
  }
});

// API Functions
async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function uploadAdvancedFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('user_id', currentUserId.toString());

  const response = await fetch(`${API_BASE_URL}/api/upload-advanced`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function getAddresses() {
  const response = await fetch(`${API_BASE_URL}/addresses`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function buildRoute(routeRequest) {
  const response = await fetch(`${API_BASE_URL}/api/route`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(routeRequest)
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function geocodeAddress(address) {
  // Простая реализация геокодирования
  // В реальном проекте здесь должен быть вызов к API геокодирования
  return {
    lat: 47.2225,
    lon: 39.7183
  };
}

async function exportReport(format) {
  const response = await fetch(`${API_BASE_URL}/api/export?format=${format}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  if (format === 'csv') {
    return await response.text();
  } else {
    return await response.json();
  }
}

// Display functions
function displayRoute(route) {
  if (!map) {
    initMap();
  }
  
  // Clear existing markers
  map.eachLayer(function(layer) {
    if (layer instanceof L.Marker) {
      map.removeLayer(layer);
    }
  });

  // Add route waypoints
  route.waypoints.forEach((waypoint, index) => {
    const marker = L.marker([waypoint.lat, waypoint.lon]).addTo(map);
    
    let popupContent = `<strong>${waypoint.address}</strong>`;
    if (waypoint.type === 'start') {
      popupContent = `<strong>Начальная точка</strong><br>${waypoint.address}`;
    } else if (waypoint.type === 'end') {
      popupContent = `<strong>Конечная точка</strong><br>${waypoint.address}`;
    } else if (waypoint.type === 'client') {
      popupContent = `<strong>Клиент #${waypoint.client_id}</strong><br>${waypoint.address}<br>Уровень: ${waypoint.client_level}`;
    }
    
    marker.bindPopup(popupContent);
    
    // Add route line
    if (index > 0) {
      const prevWaypoint = route.waypoints[index - 1];
      L.polyline([
        [prevWaypoint.lat, prevWaypoint.lon],
        [waypoint.lat, waypoint.lon]
      ], {color: '#2E8B57', weight: 3}).addTo(map);
    }
  });

  // Fit map to show all waypoints
  if (route.waypoints.length > 0) {
    const group = new L.featureGroup();
    route.waypoints.forEach(waypoint => {
      group.addLayer(L.marker([waypoint.lat, waypoint.lon]));
    });
    map.fitBounds(group.getBounds().pad(0.1));
  }
}

// Функция для отображения таблицы по умолчанию
function showDefaultTable() {
  const tbody = document.querySelector('table tbody');
  tbody.innerHTML = `
    <tr>
      <td>г. Ростов-на-Дону, ул.<br>Большая Садовая, д.<br>1</td>
      <td>Да</td>
      <td>8:15</td>
      <td>
        <input type="checkbox" name="isVisits" checked class="form-check-input" />
      </td>
    </tr>
    <tr>
      <td>г. Ростов-на-Дону, ул.<br>Большая Садовая, д.<br>1</td>
      <td>Да</td>
      <td>8:15</td>
      <td>
        <input type="checkbox" name="isVisits" checked class="form-check-input" />
      </td>
    </tr>
  `;
  console.log("Отображена таблица по умолчанию");
}

function updateTable(locations) {
  const tbody = document.querySelector('table tbody');
  tbody.innerHTML = '';

  locations.forEach((location, index) => {
    const row = document.createElement('tr');
    
    // Проверяем is_active более надежным способом
    const isChecked = (location.is_active === true || location.is_active === 't' || location.is_active === 'true' || location.is_active === 1);
    
    row.innerHTML = `
      <td>${location.address}</td>
      <td>${location.client_level === 'VIP' ? 'Да' : 'Нет'}</td>
      <td>${index + 1}:00</td>
      <td>
        <input type="checkbox" name="isVisits" class="form-check-input" 
               data-location-id="${location.location_id}" 
               ${isChecked ? 'checked' : ''} />
      </td>
    `;
    tbody.appendChild(row);
  });
}

function initMap() {
  // Initialize Leaflet map
  map = L.map('map').setView([47.2225, 39.7183], 10);
  
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);
}

// Additional API functions for user management
async function registerUser(userData) {
  const response = await fetch(`${API_BASE_URL}/api/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userData)
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function loginUser(loginData) {
  const response = await fetch(`${API_BASE_URL}/api/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(loginData)
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function getUserLocations(userId) {
  const response = await fetch(`${API_BASE_URL}/api/user/${userId}/locations`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function markLocationVisited(userId, locationId) {
  const response = await fetch(`${API_BASE_URL}/api/location/visit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      location_id: locationId
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function setStayPeriod(userId, days) {
  const response = await fetch(`${API_BASE_URL}/api/stay-period`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      days: days
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function getStayPeriod(userId) {
  const response = await fetch(`${API_BASE_URL}/api/stay-period/${userId}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

async function deleteUserLocations(userId) {
  const response = await fetch(`${API_BASE_URL}/api/user/${userId}/locations`, {
    method: 'DELETE'
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

// Add event listeners for visit checkboxes
document.addEventListener('change', function(e) {
  if (e.target.name === 'isVisits') {
    const locationId = parseInt(e.target.dataset.locationId);
    markLocationVisited(currentUserId, locationId)
      .then(result => {
        console.log('Статус посещения обновлен:', result);
        // Обновляем таблицу после успешного изменения статуса
        getUserLocations(currentUserId).then(locations => {
          if (locations.locations && locations.locations.length > 0) {
            updateTable(locations.locations);
          }
        });
      })
      .catch(error => {
        console.error('Ошибка обновления статуса:', error);
        // Возвращаем чекбокс в исходное состояние
        e.target.checked = !e.target.checked;
      });
  }
});

// Helper functions for route calculation
function calculateTotalDistance(waypoints) {
  let totalDistance = 0;
  for (let i = 0; i < waypoints.length - 1; i++) {
    const distance = calculateDistance(
      waypoints[i].lat, waypoints[i].lon,
      waypoints[i + 1].lat, waypoints[i + 1].lon
    );
    totalDistance += distance;
  }
  return Math.round(totalDistance * 100) / 100;
}

function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth's radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

function calculateTotalTime(distanceKm) {
  // Assume average speed of 30 km/h in city
  return Math.round((distanceKm / 30) * 100) / 100;
}

// Load initial data on page load
async function loadInitialData() {
  try {
    console.log(`Загружаем данные для пользователя ${currentUserId}`);
    const locations = await getUserLocations(currentUserId);
    console.log("Полученные локации при загрузке:", locations);
    
    if (locations.locations && locations.locations.length > 0) {
      updateTable(locations.locations);
      console.log(`Загружено ${locations.total_locations} локаций для пользователя ${currentUserId}`);
    } else {
      console.log("Нет локаций для пользователя", currentUserId);
      // Показываем таблицу по умолчанию если нет данных
      showDefaultTable();
    }
  } catch (error) {
    console.error('Ошибка загрузки данных:', error);
    // Показываем таблицу по умолчанию при ошибке
    showDefaultTable();
  }
}

// User management functions
function updateUserInterface() {
  const loginBtn = document.getElementById('loginBtn');
  const registerBtn = document.getElementById('registerBtn');
  const userInfo = document.getElementById('userInfo');
  const userName = document.getElementById('userName');
  const authWarning = document.getElementById('authWarning');
  
  if (currentUser) {
    loginBtn.classList.add('d-none');
    registerBtn.classList.add('d-none');
    userInfo.classList.remove('d-none');
    userName.textContent = `${currentUser.first_name} ${currentUser.last_name}`;
    if (authWarning) authWarning.classList.add('d-none');
  } else {
    loginBtn.classList.remove('d-none');
    registerBtn.classList.remove('d-none');
    userInfo.classList.add('d-none');
    if (authWarning) authWarning.classList.remove('d-none');
  }
}

function logout() {
  currentUser = null;
  currentUserId = 1; // Reset to default
  updateUserInterface();
  alert('Вы вышли из системы');
  
  // Clear forms
  document.getElementById('loginForm').reset();
  document.getElementById('registerForm').reset();
  
  // Показываем таблицу по умолчанию при выходе
  showDefaultTable();
}

function checkUserSession() {
  // Check if user is logged in (you can implement localStorage/sessionStorage here)
  // For now, we'll use default user
  updateUserInterface();
}

// Notification system
function showNotification(message, type = 'info') {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} alert-dismissible fade show position-fixed`;
  notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
  
  notification.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  
  document.body.appendChild(notification);
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.parentNode.removeChild(notification);
    }
  }, 5000);
}

// Authentication handlers
document.getElementById('loginBtn').addEventListener('click', function() {
  const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
  loginModal.show();
});

document.getElementById('registerBtn').addEventListener('click', function() {
  const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
  registerModal.show();
});

document.getElementById('logoutBtn').addEventListener('click', function() {
  logout();
});

// Login form handler
document.getElementById('loginForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  
  const phone = document.getElementById('loginPhone').value;
  const password = document.getElementById('loginPassword').value;
  
  try {
    const result = await loginUser({ phone_number: phone, password: password });
    currentUser = result;
    currentUserId = result.user_id;
    
    // Update UI
    updateUserInterface();
    
    // Close modal
    const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
    loginModal.hide();
    
    // Show success message
    showNotification('Успешный вход в систему!', 'success');
    
    // Reload data for current user
    loadInitialData();
    
  } catch (error) {
    showNotification('Ошибка входа: ' + error.message, 'error');
  }
});

// Register form handler
document.getElementById('registerForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  
  const firstName = document.getElementById('regFirstName').value;
  const lastName = document.getElementById('regLastName').value;
  const phone = document.getElementById('regPhone').value;
  const password = document.getElementById('regPassword').value;
  
  try {
    const result = await registerUser({
      first_name: firstName,
      last_name: lastName,
      phone_number: phone,
      password: password
    });
    
    // Auto-login after registration
    const loginResult = await loginUser({ phone_number: phone, password: password });
    currentUser = loginResult;
    currentUserId = loginResult.user_id;
    
    // Update UI
    updateUserInterface();
    
    // Close modal
    const registerModal = bootstrap.Modal.getInstance(document.getElementById('registerModal'));
    registerModal.hide();
    
    // Show success message
    showNotification('Регистрация успешна! Добро пожаловать!', 'success');
    
    // Reload data for current user
    loadInitialData();
    
  } catch (error) {
    showNotification('Ошибка регистрации: ' + error.message, 'error');
  }
});

// Initialize map and load data on page load
document.addEventListener('DOMContentLoaded', function() {
  initMap();
  loadInitialData();
  checkUserSession();
});
