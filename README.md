# Simple roadline detection with OpenCV

**keywords**: roadlane detection, lane detection, road line, line recognition, straight line finder

## Задача:
Создать алгоритм, который находит и выделяет линии на дороге. Отдельная линия должна быть обозначена отдельным массивом точек, т.е. если три линии, то функция возвращает массив из трех линий.

Для базового решения подходит комбинация методов OpenCV с небольшой настройкой параметров.

Основная работа проходит на файле data/road.jpeg 
Пример результата, который можно получить: result.png

Стоит проверить свой алгоритм и на других файлах из datа, сделать свой алгоритм более устойчивым и универсальным для различных типов данных. (Не стоит ожидать, что он будет очень хорошо работать на предложенных данных, но получить хоть какой-то результат на них будет хорошо.)

Приветствуются любые методы любой мощности и ресурсоемкости, использование сторонних библиотек по желанию.


## Критерии оценки:
- Верно ли найден (нагуглен?) метод и подход, рассмотрены ли альтернативы
- Наличие кода и визуализаций в jupyter notebook, демонстрирующих процесс поиска подхода
- Качество работы на файле road.jpeg (должно быть сопоставимо с result.jpeg)
- Качество работы на остальных файлах (Хорошо, если там будет "хоть что-то" найдено)