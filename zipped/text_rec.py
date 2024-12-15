import cv2

def segment_characters(thresh_name):
    contours, _ = cv2.findContours(thresh_name, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            char_images.append(thresh_name[y:y+h, x:x+w])
    char_images = sorted(char_images, key=lambda img: cv2.boundingRect(img)[0])
    return char_images

def match_character(char_img, templates):
    max_score = 0
    best_match = None
    for char, template in templates.items():
        resized_char = cv2.resize(char_img, (template.shape[1], template.shape[0]))
        result = cv2.matchTemplate(resized_char, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > max_score:
            max_score = score
            best_match = char
    return best_match

def reconstruct_name(char_images, templates):
    name = ""
    for char_img in char_images:
        char = match_character(char_img, templates)
        if char:
            name += char
    return name

def extract_name_without_ocr(name_region, templates):
    # Preprocessing
    name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    _, thresh_name = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded Name Region', thresh_name)
    cv2.waitKey(0)

    # Segment Characters
    char_images = segment_characters(thresh_name)
    print(f"Number of characters detected: {len(char_images)}")

    # Display each segmented character
    for i, char_img in enumerate(char_images):
        cv2.imshow(f"Character {i}", char_img)
        cv2.waitKey(0)

    # Reconstruct Name
    name = reconstruct_name(char_images, templates)
    print(f"Reconstructed Name: {name}")

    return name

