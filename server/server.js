const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');
const sys = require('sys')

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true })); 

app.get('/', (request, response) => {response.sendFile(path.join(__dirname, '../public', 'index.html'))});
app.use(express.static('../public'))
app.use(express.static("."));

app.listen(3000, function() { 
    console.log('server running on port 3000'); 
} ) 

app.post('/getbounds', pythonRun); 

// Runs the python file where the algoritm is
function pythonRun(req, res) {
    var bounds = req.body.mapBounds
    var points
    var finalPoints
    var pointsRansac
    var pointsCoreset
    var isBothOption = false
    var spawn = require("child_process").spawn; 
    console.log("server get the req")
    var process = spawn('python',["./googlemapskmeans/GoogleMapsKMeans/main.py", 
            bounds.south, bounds.west, bounds.north, bounds.east, req.body.Algorithm] ); 
  
    // Takes stdout data from script which executed 
    // with arguments and send this data to res object 
    process.stdout.on('data', function(data) { 
        points = data.toString().replace(/\[/g , ' ')
        points = points.replace(/\]/g,' ')
        for(i=0;i<points.length;i++){
            if(points[i]=='$'){
                points = points.split('$')
                pointsRansac = points[0].split(/\s+/)
                pointsCoreset = points[1].split(/\s+/)
                isBothOption = true
                points = [pointsRansac, pointsCoreset]
            }
        }
        if(!isBothOption){
            points = points.split(/\s+/)
        }
        res.send(points); 
    });
    process.stderr.on('data', function(data) {
            console.log('Error: ' + data);            
          });
} 
  
