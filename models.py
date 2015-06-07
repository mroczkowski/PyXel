# File: models.py
# Author: Georgiana Ogrean
# Created on 03.22.2015
#
# Change log:
#    03.22.2015 - Added simple beta-model.

# Beta-model
def betamodel(r,s0,rc,beta,bkg):
   return bkg + s0*(1.+(r/rc)**2.)**(-3.*beta+0.5)


   
