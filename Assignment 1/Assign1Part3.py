from datetime import datetime

print("Enter the dates in dd/mm/yy")
curDateStr = input("Enter current date : ")
yourBDStr = input("Enter your birth date : ")

curDate = datetime.strptime(curDateStr, "%d/%m/%y")
yourBD = datetime.strptime(yourBDStr, "%d/%m/%y")

diff = curDate - yourBD
print("Your age is {} years.".format(diff.days // 365))
