Для запуска бота:

```commandline
docker-compose up -d
```

Для логов:
```commandline
docker-compose logs -f
```

Остановить бота, сохраняя контейнер:
```commandline
docker-compose stop
```

Запустить остановленный контейнер:
```commandline
docker-compose start
```

Остановить и удалить контейнер (данные в томах сохранятся):
```commandline
docker-compose down
```

Полностью перезапустить бота с пересборкой образа:
```commandline
docker-compose up -d --build
```
