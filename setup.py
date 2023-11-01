import setuptools

#Si tienes un readme
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='FFEM',  #nombre del paquete
     version='0.1', #versión
     scripts=['FFEM.py'] , #nombre del ejecutable
     author="Jorge Felix Martinez", #autor
     author_email="jorgito16040@gmail.com", #email
     description="Easy and Fast Facial Emotion Monitoring", #Breve descripción
     long_description=long_description,
     long_description_content_type="text/markdown", #Incluir el README.md si lo has creado
     url="https://github.com/WiseGeorge/Fast-Facial-Emotion-Monitoring-FFEM-Package", #url donde se encuentra tu paquete en Github
     packages=setuptools.find_packages(), #buscamos todas las dependecias necesarias para que tu paquete funcione (por ejemplo numpy, scipy, etc.)
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
) 
