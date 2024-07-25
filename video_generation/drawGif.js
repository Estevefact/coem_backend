let drawingArea = {
    minX: 0,
    maxX: 1080,
    minY: 0,
    maxY: 1080,
};

let maxParticleSpeed = 0.3;
let fps = 25;
let revealTime = 49; //seconds
let gifDuration = 49;
let revealFrame = revealTime * fps; //seconds

// DEFAULT VALUES, DON'T CHANGE
let particles = []; // Array of particles
let img; // Image to load
let t = 0;

let scaleX = 0.5;
let scaleY = 0.5;
let autor = "";

function preload() {
    // Cargar la imagen
    autores = [
        'asuncion silva',
        'benedetti',
        'borges',
        'bukowski',
        'chejov',
        'cortazar',
        'gabo',
        'monterroso',
        'poe',
        'quiroga',
        'rulfo',
        'sagan'
    ];

    // Pick a random author
    autor = autores[Math.floor(Math.random() * autores.length)];
    // autor = autores[ 0 ]

    img = loadImage(`imgs/Pessoa.png`);
}

function createNParticles(n) {
    for (let i = 0; i < n; i++) {
        particles.push(
            new Particle(
                random(drawingArea.minX, drawingArea.maxX),
                random(drawingArea.minY, drawingArea.maxY)
            )
        );
    }
}

function setup() {
    createCanvas(drawingArea.maxX, drawingArea.maxY);
    img.loadPixels();
    particles = [];
    t = 0;
    scaleX = width / img.width;
    scaleY = height / img.height;
    // Initialize particles
    createNParticles(300);

    background(255);

    // Set frame rate
    frameRate(fps);
    console.log(autor);
}

function draw() {
    console.log("a")
    t += 1;

    if (t < revealFrame) {
        for (let particle of particles) {
            particle.update();
            particle.show();
        }

        if (particles.length < 1000) {
            // Add new particles to the simulation
            createNParticles(100);
        }
    } else {
        tint(255, t - revealFrame);
        image(img, 0, 0, width, height);
    }

    scaleX = width / img.width;
    scaleY = height / img.height;

    // Save the canvas as an image file
    if (frameCount <= gifDuration * fps) {
        saveCanvas(`gif/frame_${nf(frameCount, 4)}.png`);
    }
}

// Particle class
class Particle {
    constructor(x, y) {
        this.pos = createVector(x, y);
        this.vel = p5.Vector.random2D();
        this.maxSpeed = maxParticleSpeed;
        this.vel.setMag(this.maxSpeed);
    }

    update() {
        this.pos.add(this.vel);

        const randAngle = -PI / 5;
        this.vel.rotate(random(-randAngle, randAngle));

        if (this.vel.mag() > this.maxSpeed) {
            this.vel.setMag(this.maxSpeed);
        }

        // Wrap around the bounds of the drawing area (500-1000,0-500)
        if (this.pos.x > drawingArea.maxX) {
            this.pos.x = drawingArea.minX;
        } else if (this.pos.x < drawingArea.minX) {
            this.pos.x = drawingArea.maxX;
        }
        if (this.pos.y > drawingArea.maxY) {
            this.pos.y = drawingArea.minY;
        } else if (this.pos.y < drawingArea.minY) {
            this.pos.y = drawingArea.maxY;
        }
    }

    show() {
        // Get the color of the pixel at the current position of the particle
        let col = this.getColor();
        stroke(col);
        strokeWeight(1);
        point(this.pos.x, this.pos.y);
    }

    getColor() {
        return img.get(
            floor(this.pos.x - drawingArea.minX) / scaleX,
            floor(this.pos.y) / scaleY
        );
    }
}