import math


def main(file):
    numerics = set()
    total = 0
    total_count = 0
    not_animals = ['tomato', 'carrot', 'corn']
    fixed_f = open("fixed.txt", "a")
    animals_f = open("animals.out", "w")
    animals_dict = dict()
    for line in open(file, 'r'):
        fixed_f.write(line.replace('zerba', 'zebra'))
        line = line.strip('\n')
        for item in line.split('%'):
            if item.isnumeric():
                numerics.add(int(item))
            else:
                total_count += 1
                if animals_dict.get(item) is None:
                    animals_dict[item] = 1
                else:
                    count = animals_dict[item]
                    animals_dict[item] = count + 1
    for key in not_animals:
        total_count -= animals_dict.pop(key)
    animals_dict['zebra'] += animals_dict.pop('zerba')
    # print(animals_dict)
    numerics = list(numerics)
    numerics.sort()
    for i in numerics:
        total += i
    # print(math.log(total, math.e))
    animals_f.write(list(numerics).__str__() + '\n')
    animals_f.write("{:.2e}".format(math.log(total, math.e)) + '\n')
    animals_f.write(animals_dict.__str__() + '\n')
    animals_f.write({'total_count': total_count}.__str__())


if __name__ == '__main__':
    file_path = 'assn1_input.txt'
    main(file_path)
