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

app.listen(3000, () => {
 console.log("Server running on port 3000");
});

// example
app.get("/url", (req, res, next) => {
    res.json(["Tony","Lisa","Michael","Ginger","Food"]);
});

let MOOD = "neutral";

app.get("/mood", (req, res, next) => {
    res.json(MOOD)
    MOOD = "neutral";
})

app.post("/mood", (req, res) => {
    console.log(req)
    var action = req.body.action;
    if (action == 'happy' || action === 'sad' || action === 'angry') {
        MOOD = action;
    }
    res.json(action)
})