var express = require("express");
var bodyParser = require("body-parser");
const path = require('path');
var cors = require('cors')

var app = express();
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies
app.use(cors())

app.set("view engine", "pug");
app.set("views", path.join(__dirname, "views"));

let PORT = process.env.PORT || 3000
let MOOD = "neutral";
let STATUS = ""

app.listen(PORT, () => {
 console.log("Server running on port 3000");
});

// example
app.get("/url", (req, res, next) => {
    res.json(["Tony","Lisa","Michael","Ginger","Food"]);
});

app.get("/robot", (req, res, next) => {
    res.json(["Tony","Lisa","Michael","Ginger","Food"]);
});

app.get("/mood", (req, res, next) => {
    res.json(MOOD)
    // MOOD = "neutral";  // uncomment make expression last for a short period
})

app.get("/internal_state", (req, res, next) => {
    res.json(STATUS)
})

app.post("/mood", (req, res) => {
    console.log(req)
    var action = req.body.action;
    if (action == 'neutral' || action == 'happy' || action === 'sad' || action === 'angry' || action === 'focused' || action === 'confused' || action === 'blink' || action === 'start_blinking' || action === 'stop_blinking') {
        MOOD = action;
    }
    res.json(action)
})

app.post("/internal_state", (req, res) => {
    console.log(req)
    var status = req.body.status;
    // if (status == 'thinking' || status == 'listening' || status === '' ) {
    //     STATUS = status;
    // }
    STATUS = status
    res.json(status)
})