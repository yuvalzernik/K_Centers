const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');
const spawn = require("child_process").spawn;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true })); 


app.get('/', (request, response) => {response.sendFile(path.join(__dirname, '../public', 'index.html'))});
app.use(express.static('../public'))
app.use(express.static("."));

app.listen(8080, ()=> console.log('listening on port 8080...'));

const pythonProcess = spawn('python',["public/GoogleMapsKMeans/main.py", arg1, arg2]);

pythonProcess.stdout.on('data', (data) => {
    // Do something with the data returned from python script
});


// https://hackernoon.com/how-i-sort-of-got-around-the-google-maps-api-results-limit-1c673e66ef36  