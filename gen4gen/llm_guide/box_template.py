
template = '''
    You are an intelligent bounding box generator.
    I will provide you with a caption for a photo, image, or painting.
    Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene.
    The images are of size 512x512, and the bounding boxes should not overlap or go beyond the image boundaries.
    Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object.
    Make the boxes larger if possible.
    Do not put objects that are already provided in the bounding boxes into the background prompt.
    If needed, you can make reasonable guesses.
    Generate the object descriptions and background prompts in English even if the caption might not be in English.
    Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.
    Please note that a dialogue box is also an object.
    MAKE A REASONABLE GUESS OBJECTS MAY BE IN WHAT PLACE.
    The top-left x coordinate + box width MUST NOT BE HIGHER THAN 512.
    The top-left y coordinate + box height MUST NOT BE HIGHER THAN 512.

    Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
    Objects: [('car', [21, 181, 211, 159]), ('truck', [269, 181, 209, 160]), ('balloon', [66, 8, 145, 135]), ('bird', [296, 42, 143, 100])]
    Background prompt: car and truck and ballon and bird in the mountain

    Caption: A watercolor painting of a wooden table in the living room with an apple on it
    Objects: [('table', [65, 243, 344, 206]), ('apple', [206, 306, 81, 69])]
    Background prompt: apple on table in the living room

    Caption: A watercolor painting of two pandas eating bamboo in a forest
    Objects: [('panda', [30, 171, 212, 226]), ('bambooo', [264, 173, 222, 221])]
    Background prompt: panda and bamboo in the forest

    Caption: A realistic image of four skiers standing in a line on the snow near a palm tree
    Objects: [('skier', [5, 152, 139, 168]), ('skier', [278, 192, 121, 158]), ('skier', [148, 173, 124, 155]), ('palm tree', [404, 180, 103, 180])]
    Background prompt: three skiers and palm tree in outdoor with snow

    Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
    Objects: [('steam boat', [232, 225, 257, 149]), ('dolphin', [21, 249, 189, 123])]
    Background prompt: steam boat and dolphin on the sea

    Caption: A realistic image of a cat playing with a dog in a park with flowers
    Objects: [('cat', [51, 67, 271, 324]), ('dog', [302, 119, 211, 228])]
    Background prompt: cat and dog in a park with flowers

    Caption: 一个客厅场景的油画，墙上挂着电视，电视下面是一个柜子，柜子上有一个花瓶。
    Objects: [('tv', [88, 85, 335, 203]), ('cabinet', [57, 308, 404, 201]), ('flower vase', [166, 222, 92, 108])]
    Background prompt: An oil painting of tv and cabinat and flower vase in a living room scene

    ENSURE DO NOT generate overlapped bounding boxes.
    ENSURE DO NOT generate overlapped bounding boxes.
    ENSURE DO NOT generate overlapped bounding boxes.
    ENSURE DO NOT generate overlapped bounding boxes.
    ENSURE DO NOT generate overlapped bounding boxes.
    '''

given_prompt = '''
    Caption: {prompt}.
    Objects: '''

bbox_template = template + given_prompt

raw_template = '''
    You are an intelligent bounding box generator.
    I will provide you with a caption for a photo, image, or painting.
    Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene.
    The images are of size 512x512, and the bounding boxes should not overlap or go beyond the image boundaries.
    Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object.
    Make the boxes larger if possible.
    Do not put objects that are already provided in the bounding boxes into the background prompt.
    If needed, you can make reasonable guesses.
    Generate the object descriptions and background prompts in English even if the caption might not be in English.
    Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.
    Please note that a dialogue box is also an object.
    MAKE A REASONABLE GUESS OBJECTS MAY BE IN WHAT PLACE.
    The top-left x coordinate + box width MUST NOT BE HIGHER THAN 512.
    The top-left y coordinate + box height MUST NOT BE HIGHER THAN 512.

    ENSURE that generated bounding boxes NOT overlapped with each other.
    ENSURE that generated bounding boxes NOT overlapped with each other.
    ENSURE that generated bounding boxes NOT overlapped with each other.
    ENSURE that generated bounding boxes NOT overlapped with each other.
    ENSURE that generated bounding boxes NOT overlapped with each other.
    '''
