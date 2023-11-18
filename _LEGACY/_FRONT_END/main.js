
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
cornerstoneTools.external.Hammer = Hammer;
cornerstoneTools.external.cornerstone = cornerstone;
cornerstoneTools.external.cornerstoneMath = cornerstoneMath;

var score = 0;
var quality = "Good"

window.oncontextmenu = function(e) {
  e.preventDefault();
  return false;
}

window.onload = function() {
  let x = new XMLHttpRequest();
  x.open("GET", "/get_next_dicom", true);
  x.send();
  x.onload = function(e) {
      let res = e.currentTarget.responseText;
      res = JSON.parse(res);
      id = res.msg;
      displayDicom(id);
  }
}

window.onkeydown = function(e) {
  // if someone presses the r key reload the same dicom from server freshly
  if (e.code === 'KeyR') {
      displayDicom(id);
  }

  // check if user presses the keys 1, 2, 3, 4, or 5 and if so add .20 to score
  if (e.code === 'Digit0') {
      // prevent default keeps the quick find from triggering in browsers
      e.preventDefault();
      score = .00;
      changeRoiColor("#fff");
      updateImageData("score", ".00")
  } else if (e.code === "Digit1") {
      // prevent default keeps the quick find from triggering in browsers
      e.preventDefault();
      score = .20;
      changeRoiColor("#fd0b0b")
      updateImageData("score", ".20")
  } else if (e.code === "Digit2") {
      e.preventDefault();
      score = .40;
      changeRoiColor("#f99209")
      updateImageData("score", ".40")
  } else if (e.code === "Digit3") {
      e.preventDefault();
      score = .60
      changeRoiColor("#ecf60c")
      updateImageData("score", ".60")
  } else if (e.code === "Digit4") {
      e.preventDefault();
      score = .80;
      changeRoiColor('#2efb0d')
      updateImageData("score", ".80")
  } else if (e.code === "Digit5") {
      e.preventDefault();
      score = 1.0;
      changeRoiColor("#2889f1")
      updateImageData("score", "1.0")
  }
  // if a user presses the x key, recenter the image
  if (e.code === "KeyX") {
      e.preventDefault();
      clearROI();
  }
   // if a user presses the z key, mark as bad or good
  if (e.code === "KeyZ") {
      e.preventDefault();
      if(quality === "Good") {
          quality = "Bad";
          clearROI();
          updateImageData("quality", "Bad")
          updateImageData("score", ".00")
      } else {
          quality = "Good";
          updateImageData("quality", "Good")
      }
  }

  // if a user presses the space or enter key, submit the roi
  if (e.code === "Space" || e.code === "Enter") {
      let toolState = cornerstoneTools.globalImageIdSpecificToolStateManager.saveToolState();

      let keys = Object.keys(toolState);
      let url;

        if (toolState[keys[0]] === undefined) {
          url = setURL(0, 0, 0, 0, id, score, quality);

        } else {
            let boxStartX = Math.floor(toolState[keys[0]].RectangleRoi.data[0].handles.start.x)
            let boxStartY = Math.floor(toolState[keys[0]].RectangleRoi.data[0].handles.start.y)
            let boxEndX = Math.floor(toolState[keys[0]].RectangleRoi.data[0].handles.end.x)
            let boxEndY = Math.floor(toolState[keys[0]].RectangleRoi.data[0].handles.end.y)
            url = setURL(boxStartX, boxStartY, boxEndX, boxEndY, id, score, quality);
        }
        let x = new XMLHttpRequest();
        x.open("GET", url, true);
        x.send();
        x.onload = function (e) {
            window.location.href = "/";
        }
  } else {
      e.preventDefault();
  }
};


function displayDicom(id) {
    let viewer = document.getElementById('viewer');
    let imageId =
        "dicomweb://"+window.location.host+"/get_dicom/?id="+id;
    cornerstoneTools.init();
    cornerstone.enable(viewer);
    cornerstone.loadImage(imageId).then(function(image) {
        console.log('Loaded', image);
        cornerstoneTools.init();
        let viewer = document.getElementById('viewer');
        cornerstone.enable(viewer);
        cornerstone.displayImage(viewer, image);
        cornerstoneTools.addTool(cornerstoneTools.RectangleRoiTool);
        cornerstoneTools.addTool(cornerstoneTools.ZoomTool);
        cornerstoneTools.addTool(cornerstoneTools.PanTool);
        cornerstoneTools.setToolActive('RectangleRoi', { mouseButtonMask:1});
        cornerstoneTools.setToolActive('Zoom', { mouseButtonMask:2});
        // pan with mouse wheel
        cornerstoneTools.setToolActive('Pan', { mouseButtonMask:4});
    });
    updateImageData("sop_id", id);
}

// function tempAlert(msg,duration) {
//     let el = document.createElement("div");
//     el.setAttribute("style","position:absolute;top:10%;left:10%;background-color:white;");
//     el.style.width = "5%";
//     el.style.textAlign = "center";
//     el.innerHTML = msg;
//     setTimeout(function(){
//         el.parentNode.removeChild(el);
//         }, duration);
//     document.body.appendChild(el);
// }

function updateImageData(id, value) {
    let tooltip = document.getElementById(id);
    tooltip.innerHTML = value;
}

function clearROI() {
    let toolstate;
    toolstate = {}
    cornerstoneTools
        .globalImageIdSpecificToolStateManager
        .restoreToolState(toolstate);
    const elem = document.getElementById('viewer');
    cornerstone.updateImage(elem);
}

function setURL(x1, y1, x2, y2, id, score, quality) {
    return window.location.protocol + "//" + window.location.host +
        "/roi?x1=" + x1 + "&y1=" + y1 + "&x2=" +
        x2 + "&y2=" + y2 + "&id=" + id + "&score=" +
        score + "&quality=" + quality;
}

function changeRoiColor(color) {
    let toolState = cornerstoneTools.globalImageIdSpecificToolStateManager.saveToolState();
    let keys = Object.keys(toolState);
    console.log(toolState);
    toolState[keys[0]].RectangleRoi.data[0].color = color;
    const elem = document.getElementById('viewer');
    cornerstone.updateImage(elem);
}
