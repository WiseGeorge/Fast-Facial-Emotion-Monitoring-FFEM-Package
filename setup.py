import setuptools

#Si tienes un readme
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='FFEM',  
     version='0.1', 
     scripts=['FFEM.py'] , 
     author="Jorge Felix Martinez", 
     author_email="jorgito16040@gmail.com", 
     description="Easy and Fast Facial Emotion Monitoring", 
     long_description=long_description,
     long_description_content_type="text/markdown", 
     url="https://github.com/WiseGeorge/Fast-Facial-Emotion-Monitoring-FFEM-Package", 
     packages=setuptools.find_packages(), 
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
) 
