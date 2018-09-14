const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true })); 


app.get('/', (request, response) => {response.sendFile(path.join(__dirname, '../public', 'index.html'))});
app.use(express.static('../public'))
app.use(express.static("."));


app.listen(3000, function() { 
    console.log('server running on port 3000'); 
} ) 

app.post('/getbounds', callName); 

function callName(req, res) { 
    var spawn = require("child_process").spawn; 
    console.log("server get the req")
    var process = spawn('python',["./googlemapskmeans/GoogleMapsKMeans/main.py", 
                            req.body.south, req.body.west, req.body.north, req.body.east] ); 
  
    // Takes stdout data from script which executed 
    // with arguments and send this data to res object 
    process.stdout.on('data', function(data) { 
        res.send(data.toString()); 
    } )
    process.stdout.on('data', (data) => {
            console.log(String.fromCharCode.apply(null, data));
        });
    process.stderr.on('data', function(data) {
            console.log('Error: ' + data);
          });
} 
  
