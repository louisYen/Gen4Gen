
template = '''
    You are an intelligent scene generator.
    I will provide you with a caption for a photo, image, or painting.
    Your task is to generate the background scene for the objects mentioned in the caption.

    Caption: A photo of a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
    Scene: garden, forest, grass
    Background: in the garden, in the forest, on the grass

    Caption: A painting of a wooden table in the living room with an apple on it
    Scene: room, kitchen, office
    Background: in the room, in the kitchen, in the office

    Caption: A watercolor painting of two pandas eating bamboo in a forest
    Scene: forest, grass, garden
    Background: in the forest, on the grass, in the garden

    Caption: An image of four skiers standing in a line on the snow near a palm tree
    Scene: snow, nationalpark, forest
    Background: on the snow, in the nationalpark, in the forest

    Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
    Scene: sea, sand, beach
    Background: on the sea, on the sand, at the beach

    Caption: A photo of a cat playing with a dog in a park with flowers
    Scene: garage, garden, forest
    Background: in the garage, in the garden, in the forest

    Caption: 一个客厅场景的油画，墙上挂着电视，电视下面是一个柜子，柜子上有一个花瓶。
    Scene: room, museum, office
    Background: in the room, in the museum, in the office

    Caption: A woman sitting on the couch with her laptop in the house.
    Scene: room, office, garden
    Background: in the room, in the office, in the garden

    Caption: A woman preparing a plate of food in the kitchen.
    Scene: kitchen, office, room
    Background: in the kitchen, in the office, in the room

    Caption: A group of basketball players walking on the court.
    Scene: court, grass, beach
    Background: on the court, on the grass, at the beach

    Caption: A bed and comforter with three different cats sitting and laying down on the bed.
    Scene: room, office, garden
    Background: in the room, in the office, in the garden

    Caption: A yellow and a green motorcycle in the back of an auditorium.
    Scene: auditorium, forest, grass
    Background: in the auditorium, in the forest, on the grass

    Caption: A sign announcing the season a people walk on the sidewalk.
    Scene: street, road, sand
    Background: at the street, on the road, on the sand

    Caption: Teddy bears are re-enacting soldiers on the beach with others looking on.
    Scene: beach, sand, nationalpark
    Background: at the beach, on the sand, in the nationalpark

    Caption: A black refrigerator in a newly decorated house.
    Scene: kitchen, room, office
    Background: in the kitchen, in the room, in the office

    Caption: People on a beach and a line of surfboards
    Scene: beach, sand, grass
    Background: at the beach, on the sand, on the grass

    Caption: A cat sitting on the hood of a parked black car in a garage.
    Scene: garage, room, garden
    Background: in the garage, in the room, in the garden
    '''

given_prompt = '''
    Caption: {prompt}.
    Scene:'''

bkg_template = template + given_prompt
