from  PIL import Image, ImageDraw
import numpy

def classify_landmarks( points):
            return {
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } 


    
def PIL2array( img):
    return numpy.array(img.getdata(),numpy.uint8).reshape(img.size[1], img.size[0], 3)

def plot( frame, face_landmarks):


    pil_image = Image.fromarray(frame)
    if len(face_landmarks) > 1:
        d = ImageDraw.Draw(pil_image, 'RGBA')

    face_landmarks = classify_landmarks(face_landmarks)

    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # # Gloss the lips
    # d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    # d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    # d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    # d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # # Sparkle the eyes
    # d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    # d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # # Apply some eyeliner
    # d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    # d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    return PIL2array(pil_image)