import os


def generate_classes_text():
    print("start to generate classes text...")

    m_text = open("F:\Projects\PycharmProjects\opencvtest\detection/trainval.txt", 'w')
    for i in range(3600):
        m_text.write(str(i) + " 1\n")
    m_text.close()


if __name__ == '__main__':
    generate_classes_text()
