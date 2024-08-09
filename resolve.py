import requests
import csv
import time

with open('eggs.csv', 'w', newline='') as out:
    writer = csv.writer(out, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    with open('assistance.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
        for row in spamreader:
            query = f'{row[0]} Bahnhof'
            url = f'https://nominatim.openstreetmap.org/search?format=jsonv2&countrycodes=de&q=\"{query}\"'
            
            print(url)

            x = requests.get(url, headers={'User-Agent': 'MOTIS Accessibility Assistance'})
            print(x.text)
            guess = x.json()

            if len(guess) >= 1:
                print(row[0], ';', guess[0]['lat'], ';', guess[0]['lon'], ';', row[1])
                writer.writerow([
                    row[0],
                    ';',
                    guess[0]['lat'],
                    ';',
                    guess[0]['lon'],
                    ';',
                    row[1]
                ])
            else:
                print(row[0], ';', 0, ';', 0, ';', row[1])
                writer.writerow([
                    row[0],
                    ';',
                    0,
                    ';',
                    0,
                    ';',
                    row[1]
                ])

            time.sleep(2)
