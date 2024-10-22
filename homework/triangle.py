import math

def check_triangle(length1:float, length2:float, length3:float) -> None:
    '''
    輸入三邊長，打印為何種三角形、面積
    '''
    if (length1 + length2 <= length3)|(length1 + length3 <= length2)|(length2 + length3 <= length1):
        print("這不是三角形的邊長")
        return     
    elif (length1 == length2 == length3):
        print("全等三角形")
    elif (math.isclose(length1**2 + length2**2, length3**2))|(math.isclose(length1**2 + length3**2, length2**2))|(math.isclose(length2**2 + length3**2, length1**2)):
        if (length1 == length2)|(length1 == length3)|(length2 == length3):
            print("等腰直角三角形")
        else:
            print("直角三角形")
    elif (length1**2 + length2**2 > length3**2)&(length1**2 + length3**2 > length2**2)&(length2**2 + length3**2 > length1**2):
        print("銳角三角形")
    else:
        print("鈍角三角形")
    
    # 海龍公式算面積
    s = (length1 + length2 + length3)/2
    area = math.sqrt(s * (s - length1) * (s - length2) * (s - length3))
    print(area)

    return 

     
check_triangle(1, 2, 3) # 不是三角形
check_triangle(1, 1, 1) # 全等三角形
check_triangle(1, 1, math.sqrt(2)) # 等腰直角三角形
check_triangle(3, 4, 5) # 直角三角形
check_triangle(2, 2, 1) # 銳角三角形
check_triangle(3, 3 ,5) # 鈍角三角形