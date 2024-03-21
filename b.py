with open("src/cities.txt", "r") as f:
    lines = f.readlines()

lines = [line.replace("\n", "") for line in lines]
lines = [l for l in lines if len(l)]
lines.sort()

# print(lines)

import string
digits = string.digits

city_to_occurences = dict()
print(len(lines))

for line in lines:
    found_digit = False
    for d in digits:
        if d in line:
            print(line)
            city, num = line.split(" ")

            if city not in city_to_occurences.keys():
                city_to_occurences[city] = int(num)
            else:
                city_to_occurences[city] += int(num)
            found_digit = True
            break
    if found_digit:
        continue
    if line not in city_to_occurences.keys():
        city_to_occurences[line] = 1
    else:
        city_to_occurences[line] += 1

print(city_to_occurences)

city_to_occ = list(city_to_occurences.items())

city_to_occ = sorted(city_to_occ, key=lambda x: (-1) * x[1])

print("*" * 50)
print(city_to_occ)

print("Name of the city  | number of times it occured  |  % of articles where it "
      "occured")
for i, (city, occ) in enumerate(city_to_occ):
    print(f"{i + 1}. {city}  |  {occ}  |  {round(occ/50 * 100)}%")

lines = ""
for i, (city, occ) in enumerate(city_to_occ):
    lines += f"""<tr>
            <td>{i + 1}</td>
            <td>{city}</td>
            <td>{occ}</td>
            <td>{round(occ/50 * 100)}</td>
        </tr>\n"""


orig = """+------+------------------------------+--------+------+
| Rank | Name of the city  | number of times it occured  |  % of articles where it 
occured |
+======+==============================+========+======+"""


lines = ""

# for i, (city, occ) in enumerate(city_to_occ):
#     lines += f"""\n| {i + 1}  | {city}       | {occ}      |{round(occ/50 * 100)} |
# +------+------------------------------+--------+------+"""

d = {"Name of the city": [], "Number of times it occurred": [],
     "% of articles where it occurred": []}


for i, (city, occ) in enumerate(city_to_occ):
    # d["Rank"].append(i + 1)
    d["Name of the city"].append(city)
    d["Number of times it occurred"].append(occ)
    d["% of articles where it occurred"].append(f"{round(occ/50 * 100)}%")

import pandas as pd
pd.DataFrame.from_dict(d).to_csv("data.csv", index=False)

orig += lines
print(orig)
exit(0)
s = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<table border="1">
    <thead>
        <tr>
            <th>Rank</th>
            <th>Name of the city</th>
            <th>Number of times it occurred</th>
            <th>% of articles where it occurred</th>
        </tr>
    </thead>
    <tbody>
    {lines}
    </tbody>
</table>

</body>
</html>
"""
print(s)
import geostring as geo
import collections

# country_to_occ = dict()
# for (city, occ) in city_to_occ:
#     result = geo.resolve(city)
#     if (
#             isinstance(result, collections.OrderedDict)
#             and "resolved_country" in result
#             and result["resolved_country"]
#     ):
#         # print(result)
#         print(result["resolved_subcountry"])
#         if result["resolved_country"] not in country_to_occ:
#             country_to_occ[result["resolved_country"]] = occ
#         else:
#             country_to_occ[result["resolved_country"]] += occ

# print(country_to_occ)

country_to_occ = [('azerbaijan', 1), ('montenegro', 1), ('serbia', 2), ('slovakia', 2),
                  ('latvia', 2), ('bulgaria', 2), ('slovenia', 3), ('iceland', 3),
                  ('georgia', 3), ('malta', 4), ('lithuania', 4), ('turkey', 5),
                  ('luxembourg', 5), ('romania', 6), ('croatia', 6), ('hungary', 6),
                  ('estonia', 9), ('greece', 9), ('norway', 9), ('czech republic', 13),
                  ('sweden', 14), ('finland', 15), ('belgium', 16), ('ireland', 17),
                  ('denmark', 21), ('italy', 24), ('austria', 25), ('poland', 27),
                  ('portugal', 35), ('netherlands', 41), ('france', 45),
                  ('switzerland', 52), ('united kingdom', 66), ('germany', 68), ('spain', 78)]

country_to_occ = [(c[0].title(), c[1]) for c in country_to_occ]
print("*" * 50)
print(country_to_occ)

print(sum([c[1] for c in country_to_occ]))

"""

22 + x = w
16 + 2 * y = w
x + y + 12 =w 

x - w = -22 => x = w - 22
2 * y - w = -16 => y = w/2 - 8
x + y - w = -12

w - 22 + w/2 - 8 - w = -12

w/2 = 18
w = 36
x = 14
y = 10

"""