@echo off
chcp 65001 >nul
echo ========================================
echo УСТАНОВКА СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ
echo ========================================
echo.

REM Проверка прав администратора
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Запуск от имени администратора...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ✅ Права администратора получены
echo.

REM Проверка Python 3.11
echo 🔍 Проверка Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3.11 не найден!
    echo.
    echo Установите Python 3.11.9 с официального сайта:
    echo https://www.python.org/downloads/release/python-3119/
    echo.
    echo Убедитесь, что установили с галочкой "Add Python to PATH"
    pause
    exit /b 1
)

echo ✅ Python 3.11 найден
py -3.11 --version
echo.

REM Удаление старого виртуального окружения
echo 🗑️  Удаление старого виртуального окружения...
if exist venv311 (
    rmdir /s /q venv311
    echo Старое окружение удалено
)

REM Создание нового виртуального окружения
echo 🏗️  Создание виртуального окружения...
py -3.11 -m venv venv311

if errorlevel 1 (
    echo ❌ Не удалось создать виртуальное окружение
    pause
    exit /b 1
)

echo ✅ Виртуальное окружение создано
echo.

REM Активация виртуального окружения
echo 🔧 Активация окружения...
call venv311\Scripts\activate.bat

if errorlevel 1 (
    echo ❌ Не удалось активировать виртуальное окружение
    pause
    exit /b 1
)

echo ✅ Окружение активировано
echo.

REM Обновление pip
echo 📦 Обновление pip...
python -m pip install --upgrade pip --quiet

if errorlevel 1 (
    echo ❌ Не удалось обновить pip
    pause
    exit /b 1
)

echo ✅ Pip обновлен
echo.

REM Установка зависимостей
echo 📚 Установка зависимостей из requirements.txt...
if not exist requirements.txt (
    echo ❌ Файл requirements.txt не найден
    echo Создаю файл requirements.txt...
    
    (
        echo face-recognition==1.3.0
        echo opencv-python==4.8.1.78
        echo numpy==1.24.3
        echo scikit-learn==1.3.0
        echo Pillow==10.0.0
        echo customtkinter==5.2.0
        echo tkinterweb==3.2.1
        echo kagglehub==0.2.3
        echo tqdm==4.66.2
        echo requests==2.31.0
    ) > requirements.txt
    
    echo ✅ Файл requirements.txt создан
)

echo Установка может занять несколько минут...
pip install -r requirements.txt

if errorlevel 1 (
    echo ⚠️  Были ошибки при установке некоторых пакетов
    echo Пробую установить по одному...
    
    echo Устанавливаю numpy...
    pip install numpy==1.24.3
    
    echo Устанавливаю opencv-python...
    pip install opencv-python==4.8.1.78
    
    echo Устанавливаю scikit-learn...
    pip install scikit-learn==1.3.0
    
    echo Устанавливаю Pillow...
    pip install Pillow==10.0.0
    
    echo Устанавливаю customtkinter...
    pip install customtkinter
    
    echo Устанавливаю tkinterweb...
    pip install tkinterweb==3.2.1
    
    echo Устанавливаю kagglehub...
    pip install kagglehub tqdm requests
    
    echo Устанавливаю face-recognition...
    pip install cmake
    pip install dlib==19.24.2
    pip install face-recognition==1.3.0
)

echo ✅ Зависимости установлены
echo.

REM Создание структуры проекта
echo 📁 Создание структуры проекта...
python -c "
import os
from config import Config
Config.setup_directories()
print('Структура проекта создана')
"

if errorlevel 1 (
    echo ⚠️  Ошибка создания структуры
    echo Создаю вручную...
    
    mkdir dataset 2>nul
    mkdir dataset\Alexander 2>nul
    mkdir dataset\Egor 2>nul
    mkdir dataset\Unknown 2>nul
    mkdir models 2>nul
    mkdir uploads 2>nul
    mkdir uploads\processed 2>nul
    mkdir results 2>nul
    mkdir results\images 2>nul
    mkdir results\videos 2>nul
    mkdir results\reports 2>nul
    
    echo ✅ Структура проекта создана вручную
)
echo.

REM Скачивание LFW датасета
echo 🌐 Скачивание LFW датасета (может занять время)...
echo Это может занять несколько минут в зависимости от скорости интернета...
python download_lfw.py

if errorlevel 1 (
    echo ⚠️  Ошибка скачивания LFW датасета
    echo Пропускаем этот шаг...
)

echo.

REM Сбор изображений из LFW
echo 🖼️  Сбор изображений из LFW датасета...
if exist collect_lfw_images.py (
    python collect_lfw_images.py
) else (
    echo ⚠️  Скрипт collect_lfw_images.py не найден
    echo Скачивание и сбор LFW пропущены
)

echo.

REM Проверка установки
echo 🔍 Проверка установки...
python -c "
try:
    import numpy as np
    import cv2
    import face_recognition
    from PIL import Image
    import customtkinter as ctk
    print('✅ Все основные библиотеки установлены корректно')
    print(f'   numpy: {np.__version__}')
    print(f'   OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ Ошибка импорта: {e}')
"

echo.
echo ========================================
echo 🎉 УСТАНОВКА ЗАВЕРШЕНА!
echo ========================================
echo.
echo 📋 ЧТО ДАЛЬШЕ:
echo.
echo 1. Захватите фото для обучения:
echo    python main.py
echo    -> Нажмите "Захватить фото Александра"
echo    -> Нажмите "Захватить фото Егора"
echo.
echo 2. Обучите модель:
echo    -> Нажмите "Обновить модель"
echo.
echo 3. Используйте распознавание:
echo    -> Нажмите "Запуск камеры" для реального времени
echo    -> Или загружайте изображения в папку uploads/
echo.
echo 📁 СТРУКТУРА ПРОЕКТА:
echo    dataset\Александр\     - фото Александра
echo    dataset\Егор\          - фото Егора
echo    dataset\Неизвестный\   - фото других людей
echo    uploads\              - для автоматической обработки
echo    results\              - результаты работы
echo.
echo 🚀 ЗАПУСК ПРОЕКТА:
echo    python main.py
echo.
echo ========================================
pause
