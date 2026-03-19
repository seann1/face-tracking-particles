import p5 from 'p5';

const sketch = (p) => {
	let video;
	// Video display rect (16:9 fitted)
	let vx, vy, vw, vh;
	let faceMesh;
	let options = { maxFaces: 1, refineLandmarks: false, flipped: false };
	let faces = [];
	
	function calcVideoRect() {
		let aspect = 16 / 9;
		if (p.width / p.height > aspect) {
			// Window is wider than 16:9 — pillarbox
			vh = p.height;
			vw = vh * aspect;
		} else {
			// Window is taller than 16:9 — letterbox
			vw = p.width;
			vh = vw / aspect;
		}
		vx = (p.width - vw) / 2;
		vy = (p.height - vh) / 2;
	}
	
	function modelLoaded() {
		console.log("Model loaded!");
	}
	
	// javascript
	function drawPolygonFromIndices(face, indices, scaleX, scaleY, holeIndices) {
	  p.push();
	  p.fill(0, 255, 0, 80);
	  p.stroke(0, 255, 0);
	  p.strokeWeight(1.5);
	  p.beginShape();
	  for (let k = 0; k < indices.length; k++) {
	    const idx = indices[k];
	    const kp = face.keypoints[idx];
	    if (!kp) continue;
	    let mappedX = kp.x * scaleX;
	    if (options.flipped) mappedX = vw - mappedX;
	    const mappedY = kp.y * scaleY;
	    p.vertex(vx + mappedX, vy + mappedY);
	  }
	
	  if (holeIndices && holeIndices.length) {
	    p.beginContour();
	    for (let k = holeIndices.length - 1; k >= 0; k--) {
	      const idx = holeIndices[k];
	      const kp = face.keypoints[idx];
	      if (!kp) continue;
	      let mappedX = kp.x * scaleX;
	      if (options.flipped) mappedX = vw - mappedX;
	      const mappedY = kp.y * scaleY;
	      p.vertex(vx + mappedX, vy + mappedY);
	    }
	    p.endContour();
	  }
	
	  p.endShape(p.CLOSE);
	  p.pop();
	}
	
	const faceOvalIndices = [
		10, 338, 297, 332, 284, 251, 389, 356,
		454, 323, 361, 288, 397, 365, 379, 378,
		400, 377, 152, 148, 176, 149, 150, 136,
		172, 58, 132, 93, 234, 127, 162, 21,
		54, 103, 67, 109
	];
	const mouthOuterIndices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61];
	const mouthInnerIndices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
	const leftEyeIndices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
	const rightEyeIndices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
	
	// ensure window resize recalculates video rect
	p.windowResized = () => {
	  p.resizeCanvas(p.windowWidth, p.windowHeight);
	  calcVideoRect();
	};
	function gotFaces(results) {
		// Save the output to the faces variable
		faces = results;
	}
	
	p.setup = async () => {
		p.createCanvas(p.windowWidth, p.windowHeight);
		video = p.createCapture(p.VIDEO);
		video.hide();
		video.size(1920, 1080);
		
		faceMesh = await ml5.faceMesh(options);
		
		calcVideoRect();
		
		await faceMesh.detectStart(video, gotFaces);
	};
	
	p.draw = () => {
		p.background(0);
		p.image(video, vx, vy, vw, vh);
		
		// determine the actual video pixel size reported by the capture element
		const srcW = (video.width && video.width > 0) ? video.width : (video.elt && video.elt.videoWidth) || 640;
		const srcH = (video.height && video.height > 0) ? video.height : (video.elt && video.elt.videoHeight) || 480;
		
		// compute scale from source video pixels to canvas rectangle
		const scaleX = vw / srcW;
		const scaleY = vh / srcH;
		
		// draw keypoints mapped into the canvas rect (and flipped if needed)
		for (let i = 0; i < faces.length; i++) {
			const face = faces[i];
			for (let j = 0; j < face.keypoints.length; j++) {
				const kp = face.keypoints[j];
				// handle optional horizontal flip (mirror)
				let mappedX = kp.x * scaleX;
				if (options.flipped) {
					mappedX = vw - mappedX;
				}
				const mappedY = kp.y * scaleY;
				
				const drawX = vx + mappedX;
				const drawY = vy + mappedY;
				
				p.fill(0, 255, 0);
				p.noStroke();
				p.circle(drawX, drawY, 5);
			}
		}
		
		console.log(faces.length);
		for (let i = 0; i < faces.length; i++) {
			const face = faces[i];
			drawPolygonFromIndices(face, faceOvalIndices, scaleX, scaleY);
			drawPolygonFromIndices(face, leftEyeIndices, scaleX, scaleY);
			drawPolygonFromIndices(face, rightEyeIndices, scaleX, scaleY);
			drawPolygonFromIndices(face, mouthOuterIndices, scaleX, scaleY);
		}
	};
	
	// javascript
	// Insert into `script.js` (near the top-level variables)
	
	
	p.windowResized = () => {
		p.resizeCanvas(p.windowWidth, p.windowHeight);
	};
};

new p5(sketch, document.getElementById('sketch-container'));





