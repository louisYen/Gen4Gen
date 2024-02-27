
template = '''
    You are an intelligent object ratio generator.
    I will provide you with several object names.
    Your task is to generate the reasonable ratio for objects in the real world.
    The ratio of the biggest object equals to 1.
    The ratio of the smallest object equals to 0.4.

    Objects: house, person, pig, cow, car
    Ratio: 1.0, 0.5, 0.40, 0.55, 0.85

    Caption: bus, car, car, phone box
    Ratio: 1.0, 0.7, 0.7, 0.6

    Objects: cat, dog
    Ratio: 0.7, 1.0

    Objects: car, restroom
    Ratio: 0.8, 1.0

    Objects: cat, dog, house plant
    Ratio: 0.8, 1.0, 0.75

    Objects: house plant, fridge
    Ratio: 0.6, 1.0
    '''

given_prompt = '''
    Objects: {prompt}.
    Ratio:'''

ratio_template = template + given_prompt
