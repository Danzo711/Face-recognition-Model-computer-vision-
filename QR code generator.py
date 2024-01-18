# import qrcode

# names = ["kunj", "vedant", "nikunj", "devarsh"]
# count=1
# with open("names.txt","w+") as file:
#     for i in names:
#         if count==len(names):
#             file.write(i)

#         else:
#            count=count+1
#            file.write(i+",")
#            print(count)



#     names = file.read().split(",")

# for name in names:
#     qr = qrcode.QRCode(
#         version=5,
#         box_size=10,
#         border=4
#     )
    
#     qr.add_data(name)
#     qr.make(fit=True)
    
#     qr_image = qr.make_image(fill_color="black", back_color="blue")
#     qr_image.save(f"{name}.png")

#     print(f"QR code generated for {name}")

import qrcode
names = ["kunj", "vedant", "nikunj", "devarsh"]
def adding_user():
    x=input("enter the name :")
   
    new_users=[]
    new_users.append(x)
def file_access():
    count = 1 
    
    # Open the "names.txt" file for writing and reading ('w+' mode).
    with open("names.txt", "w+") as file:
        # Iterate through each name in the 'names' list.
        for i in names:
            # Write the current name to the file.
            file.write(i)
            if count < len(names):
                file.write(",")
            count += 1
        # names_1 = file.read().split(",")
def generator():
    for name in names:
        qr = qrcode.QRCode(
            version=5, # size of the QRCode
            box_size=10, # box size of the QRCode
            border=4 # denotes the width of border of the QRCode
        )

        qr.add_data(name) # add data to the QRCode using qr object
        qr.make(fit=True)

        qr_image = qr.make_image(fill_color="red", back_color="yellow") # set colours
        qr_image.save(f"{name}.png") # save QRCode image

        print(f"QR code generated for {name}")

def main():

    print("select any of the choices")
    x=input("Do you want to add a user?\n")
    if x=="1":
        adding_user()
    elif x=="2":
        exit
    
if __name__=="__main__":
    main()
    file_access()
    generator()
    