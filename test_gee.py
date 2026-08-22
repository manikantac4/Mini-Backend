import ee

ee.Authenticate()
ee.Initialize(project="water-segmentation-gee")

print("Connected")