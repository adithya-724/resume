// Particle canvas
(function(){
  const c=document.getElementById('bg-canvas'),ctx=c.getContext('2d');
  let w,h,particles=[];
  function resize(){w=c.width=innerWidth;h=c.height=innerHeight}
  resize();window.addEventListener('resize',resize);
  class P{
    constructor(){this.reset()}
    reset(){this.x=Math.random()*w;this.y=Math.random()*h;this.vx=(Math.random()-.5)*.3;this.vy=(Math.random()-.5)*.3;this.r=Math.random()*1.5+.5;this.a=Math.random()*.5+.1}
    update(){this.x+=this.vx;this.y+=this.vy;if(this.x<0||this.x>w)this.vx*=-1;if(this.y<0||this.y>h)this.vy*=-1}
    draw(){ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle=`rgba(6,182,212,${this.a})`;ctx.fill()}
  }
  const count=Math.min(80,Math.floor(w*h/15000));
  for(let i=0;i<count;i++)particles.push(new P());
  function frame(){
    ctx.clearRect(0,0,w,h);
    for(let i=0;i<particles.length;i++){
      particles[i].update();particles[i].draw();
      for(let j=i+1;j<particles.length;j++){
        const dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y,d=dx*dx+dy*dy;
        if(d<18000){ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);ctx.lineTo(particles[j].x,particles[j].y);ctx.strokeStyle=`rgba(6,182,212,${.08*(1-d/18000)})`;ctx.stroke()}
      }
    }
    requestAnimationFrame(frame);
  }
  frame();
})();

// Nav scroll
const nav=document.getElementById('navbar');
window.addEventListener('scroll',()=>{nav.classList.toggle('scrolled',scrollY>50)});

// Mobile nav with hamburger animation
function toggleNav(){
  const navLinks=document.getElementById('navLinks');
  const hamburger=document.getElementById('hamburger');
  navLinks.classList.toggle('open');
  hamburger.classList.toggle('open');
}
function closeNav(){
  const navLinks=document.getElementById('navLinks');
  const hamburger=document.getElementById('hamburger');
  navLinks.classList.remove('open');
  hamburger.classList.remove('open');
}

// Intersection observer for reveals
const observer=new IntersectionObserver((entries)=>{entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');
  // also reveal timeline items inside
  e.target.querySelectorAll('.timeline-item').forEach((t,i)=>{setTimeout(()=>t.classList.add('visible'),i*150)});
}})},{threshold:.15});
document.querySelectorAll('.reveal').forEach(el=>observer.observe(el));
