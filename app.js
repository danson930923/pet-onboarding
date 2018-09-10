/*After Page Load*/

function OnLoadFunctions(){
	let figureTitle = document.getElementById("FigureTitle");
	figureTitle.style.opacity = 1;
}

/*Animation related to Header*/
const header = document.querySelector('header');
header.addEventListener("click", (e) => {
	const navListElements = [
		{"Name":"WorkList", "LabelDOM":document.getElementById('WorkLabel'), "ListDOM":document.getElementById('WorkList')},
		{"Name":"StudyList", "LabelDOM":document.getElementById('StudyLabel'), "ListDOM":document.getElementById('StudyList')},
		{"Name":"TravelList", "LabelDOM":document.getElementById('TravelLabel'), "ListDOM":document.getElementById('TravelList')}
	]

	/*Remove all clicked class && Add to specific one*/
	document.getElementById('HomeLabel').classList.remove("a-click");
	document.getElementById('ContactLabel').classList.remove("a-click");
	navListElements.map((ele) => {
		ele.LabelDOM.classList.remove("a-click");
	})
	e.target.classList.add("a-click");

	/*Calculate list postiion*/
	switch (e.target.id){
		case 'WorkLabel':
			e.target.classList.add("a-click");
			navListElements.map((e) => {
				e.Name === "WorkList" ? e.ListDOM.style.display = 'block' : e.ListDOM.style.display = 'none';
				if (e.Name === "WorkList") {
					e.ListDOM.style.left = e.LabelDOM.offsetLeft + 'px';
					e.ListDOM.style.top = Math.floor(header.querySelector('.nav-label').offsetHeight) + 'px';
				}
			})
			break;
		case 'StudyLabel':
			navListElements.map((e) => {
				e.Name === "StudyList" ? e.ListDOM.style.display = 'block' : e.ListDOM.style.display = 'none';
				if (e.Name === "StudyList") {
					e.ListDOM.style.left = e.LabelDOM.offsetLeft + 'px';
					e.ListDOM.style.top = Math.floor(header.querySelector('.nav-label').offsetHeight) + 'px';
				}
			})
			break;
		case 'TravelLabel':
			navListElements.map((e) => {
				e.Name === "TravelList" ? e.ListDOM.style.display = 'block' : e.ListDOM.style.display = 'none';
				if (e.Name === "TravelList") {
					e.ListDOM.style.left = e.LabelDOM.offsetLeft + 'px';
					e.ListDOM.style.top = Math.floor(header.querySelector('.nav-label').offsetHeight) + 'px';
				}
			})
			break;
		default:
			navListElements.map((e) => {
				e.ListDOM.style.display = 'none';
			})
	}

	/*Mouse Leave List Event*/
	for (let i = 0; i < navListElements.length; i++){
		navListElements[i].ListDOM.addEventListener("mouseleave", (event) => {
			navListElements[i].ListDOM.style.display = 'none';
			e.target.classList.remove("a-click");
		});
	}
});


/*Animation related to Figure*/
let figureDownBar = document.getElementById("FigureDownBar");
figureDownBar.addEventListener("mouseover", () => {
	figureDownBar.style.opacity = 1;
});
figureDownBar.addEventListener("mouseleave", () => {
	figureDownBar.style.opacity = 0;
});
figureDownBar.addEventListener("click", () => {
	document.querySelector('#Blockquote').scrollIntoView({ 
	  behavior: 'smooth' 
	});
});