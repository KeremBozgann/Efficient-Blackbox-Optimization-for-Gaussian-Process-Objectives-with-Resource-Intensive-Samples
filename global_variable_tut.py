
def main_wrap():

    def main():

        def f1():

            global x

            x= 3

            y= 2*x

            return y

        def f2():

            # global x
            z= x**2

            return z

        y= f1()

        z= f2()

        print(y, z)

    print(x)