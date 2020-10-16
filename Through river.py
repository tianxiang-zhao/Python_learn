print("野人的个数 船的容量")
n=input()
n=int(n)
ship=input()
ship=int(ship)
if(ship%2==0):
    ship=ship
else:
    ship=ship-1
right_a=int(n)#表示教士的个数
right_b=int(n)#表示野人的个数
left_a=0
left_b=0
def goto_right(m,n,o):
    print("载",m,"教士",n+o,"野人去右岸")

def goto_left(m,n,o):
    print("载", m, "教士", n+o, "野人返回左岸")


while(1):
    if (right_a == 0):
        break
    if((right_a==right_b)and(left_b==0)):#第一次出发
        goto_right(0,ship,0)
        goto_left(0,0,ship/2)
        left_b=left_b+ship/2
        right_b=right_b-ship
        print("左岸", right_a, right_b, "\t右岸", left_a, left_b)

    if((right_a==right_b)and(left_b!=0)):#载野人去右岸
        goto_right(0,ship/2,ship/2)
        left_b=left_b+ship/2
        right_b=right_b-ship/2
        if (right_a > 0):
            goto_left(0, 0, ship / 2)
        print("左岸", right_a, right_b, "\t右岸", left_a, left_b)

    if(right_a>right_b):       #载教士去右岸
        goto_right(ship/2,0,ship/2)
        left_a=left_a+ship/2
        right_a=right_a-ship/2
        if (right_a > 0):
            goto_left(0, 0, ship / 2)
        print("左岸", right_a, right_b, "\t右岸", left_a, left_b)


    if((ship/2>right_b)and(right_b!=0)):
        remain=ship/2-right_b
        ship=right_b*2
        print("多余的野人下船")
        left_b=left_b+remain
        print("左岸", right_a, right_b, "\t右岸", left_a, left_b)

    if (right_a == 0):
        left_b = left_b + ship / 2
        print("船上的野人下船")
        print("左岸", right_a, right_b, "\t右岸", left_a, left_b)
    if((left_b>left_a)or(right_b>right_a)):
        print("渡河失败！！")
        break