#!/usr/bin/env python

from fp_site import app
from secret_key import *

app.secret_key = secret_key
app.run(debug = True)

