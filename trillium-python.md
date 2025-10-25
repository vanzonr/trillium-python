---
title: Python on Trillium and Open OnDemand
author: Ramses van Zon
date: October 27, 2025
---

# Introduction

## In this workshop...

 * Why Python
 
 * Why Supercomputers

 * Compartimentalize

 * OnDemand

# Why Python

## Python is great

Python is usually the 'driver' for AI and ML.

Its greatness is in large part due to the available packages.

And it in interactive computing paradigm
(enter Jupyter Lab)

# Why Supercomputers

## I want more

You may need more resources than your laptop can provide.

So you go to one of the Alliance's 'advanced research computing' clusters, like Nibi, Fir, Narval, Rorqual and Trillium.

So what?

Well, using a supercomputer is to a whole nother thing!

## Sharing is caring

* It is a shared resource.

* It is remote.

* It's usually command-line driven

* It is not your own machine.

## So what?

Unfortunately, many tutorials on python, ai and ml assume that your working on your own machine and have full privileges to reconfigure it (and mess it up).

This afternoon, we'll show you how to operate in this shared space, focussing in particular on Trillium (but touching on the others as well).

# Get access

## Let's get on an account!

What do you need to follow along this afternoon:

- An account on Trillium
- which you get by having an account on CCDB
- requesting access
- and setup MFA
- Optional for today: setup SSH keys.

https://scinethpc.ca/getting-started/

This will give you access both Trillium and SciNet's OnDemand service.

## Resources

### What is what?

SciNet's OnDemand is a web interface to Trillium, and a bit more.

## Trillium is a compute cluster

Overview of trillium.

## Restrictions of using trillium

which you don't have on GPs

also:
- no GUI
- no Desktop
- no VSCode (caveats...)

What do you gain:
- larger computations run much easier
- file system is fantastic
- full H100 for heavy GPU and AI workloads.

# On Demand

## Not everything needs 192 cores

But what if you have that one postprocessing step that you need less than 192 cores for?  What if you need to do some visualization?
For interactive work of that and other kinds in python, JupyterLab is typically used.

We installed the OnDemand to provide this JL and other features in the browser.

## What is OnDemand?

Let's jump in (hands on)

In your browser, log into https://ondemand.scinet.utoronto.ca

Use your CCDB account
Use your CCDB password
Use your MFA

You'll see the ondemand interface.

OnDemand is this web interface. It was developed at OSC, and is getting widely adopted for many supercomputing systems. In Canada, Trillium, Nibi, and Vulcan, as well as on Grex 

# More

## Introduction to JupyterLab

## Best practices

## Notebooks (Python, R, Julia)

## Virtual Desktop, VS Code, OpenRefine, LibreQDA

## Resource Monitoring
